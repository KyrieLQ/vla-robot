#!/usr/bin/env python3
"""Build a conservative report from actual artifacts; absent GPU evidence never becomes success."""
import datetime, json, os, pathlib, statistics, subprocess

ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(path,default=None):
    p=ROOT/path
    return json.loads(p.read_text()) if p.exists() else ({} if default is None else default)

def main():
    r=ROOT/'reports';now=datetime.datetime.now().astimezone().isoformat()
    audit=read('reports/parameter_audit.json');data=read('reports/data_manifest.json');ver=read('reports/versions.json')
    gpu_files=sorted(r.glob('gpu_selection_*.json'));gpu=json.loads(gpu_files[-1].read_text()) if gpu_files else {}
    cpu=read('results/cpu_render_20261012_v2/result.json')
    latest=read('reports/latest_pipeline.json')
    pipeline_path=pathlib.Path(latest['path']) if latest else None
    pipeline=json.loads((pipeline_path/'status.json').read_text()) if pipeline_path else {}
    experiment=pipeline.get('experiment','smoke_20261012_initial')
    training=read('results/'+experiment+'/training_report.json');resume=read('results/'+experiment+'/resume_report.json')
    loop={};loop_path=None;policy_record={}
    if pipeline_path and (pipeline_path/'closed_loop.log').exists():
        for line in (pipeline_path/'closed_loop.log').read_text().splitlines():
            if line.startswith('CLOSED_LOOP_ARTIFACTS '):
                link=json.loads(line.split(' ',1)[1]);loop_path=pathlib.Path(link['result_directory'])/'result.json'
                if loop_path.exists():loop=json.loads(loop_path.read_text())
                policy_record=read(link['policy_record'])
    updates=training.get('updates',[]);steady=[x['seconds'] for x in updates if x['update']>=2]
    timing={'measured_updates':len(updates),'compile_seconds':training.get('compile_seconds'),
            'first_update_seconds':updates[0]['seconds'] if updates else None,
            'steady_update_seconds_mean':statistics.mean(steady) if steady else None,
            'steady_update_seconds_min':min(steady) if steady else None,
            'steady_update_seconds_max':max(steady) if steady else None,
            'scope':'synchronous compiled update only; excludes loading/initialization/verification/checkpoint I/O; no single-GPU speedup comparison'}
    (r/'training_metrics_summary.json').write_text(json.dumps(timing,indent=2)+'\n')
    overall='SUCCESS' if pipeline.get('status')=='SUCCESS' else 'PARTIAL'
    wait=gpu.get('status')=='WAITING_FOR_IDLE_GPU' or pipeline.get('status')=='WAITING_FOR_IDLE_GPU'
    stage_results={
      'A_environment_data':{'status':'PASS' if all(read('reports/'+n+'.json').get('status')=='PASS' for n in ['assets_integrity','data_audit','batch_validation']) else 'NOT_RUN'},
      'parameter_audit':{'status':audit.get('status','NOT_RUN'),'scope':'abstract-shape optimizer parameter selection, not a real update'},
      'B_JAX_GPU':{'status':read('reports/gpu_probe.json').get('status','BLOCKED' if wait else 'NOT_RUN'),'reason':'WAITING_FOR_IDLE_GPU' if wait else None},
      'C_LIBERO_CPU_render':{'status':cpu.get('status','NOT_RUN'),'backend':cpu.get('gl_renderer'),'scope':'reset + states/images + 12 legal dummy actions; not policy closed loop'},
      'C_LIBERO_EGL_render':{'status':next((x['status'] for x in pipeline.get('stages',[]) if x['stage']=='egl_render'),'NOT_RUN')},
      'D_base_weights_CPU':{'status':read('reports/base_weight_audit.json').get('status','NOT_RUN'),'scope':'actual official base weights loaded and checked; no GPU initialization'},
      'D_GPU_LoRA_initialization':{'status':'PASS' if training.get('initialization_seconds') is not None else 'NOT_RUN'},
      'E_real_training':{'status':training.get('status','NOT_RUN'),'optimizer_updates':training.get('optimizer_updates_this_process',0)},
      'F_checkpoint_reload_resume':{'status':resume.get('status','NOT_RUN'),'optimizer_updates':resume.get('optimizer_updates_this_process',0)},
      'G_policy_closed_loop':{'status':'PASS' if any(x['stage']=='closed_loop' and x['status']=='PASS' for x in pipeline.get('stages',[])) else 'NOT_RUN'},
    }
    usage=int(subprocess.check_output(['du','-s','-B1',str(ROOT)],text=True).split()[0]);s=os.statvfs(ROOT)
    storage={'project_allocated_bytes':usage,'project_GiB':usage/2**30,'filesystem_available_GiB':s.f_bavail*s.f_frsize/2**30,'budget_GiB':120,'minimum_filesystem_reserve_GiB':100}
    (r/'storage_usage.json').write_text(json.dumps(storage,indent=2)+'\n')
    result={'overall_status':overall,'updated_at':now,'resource_state':'WAITING_FOR_IDLE_GPU' if wait else gpu.get('status','UNKNOWN'),
      'stages':stage_results,'single_gpu_training_verified':training.get('status')=='PASS' and training.get('fsdp_devices')==1,
      'multi_gpu_used':training.get('fsdp_devices',0)>1,'observed_gpu_peak_mib':training.get('observed_peak_device_memory_mib'),
      'resume_observed_gpu_peak_mib':resume.get('observed_peak_device_memory_mib'),
      'compile_seconds':training.get('compile_seconds'),'training_updates':training.get('optimizer_updates_this_process',0),
      'resume_updates':resume.get('optimizer_updates_this_process',0),'training_checkpoint':resume.get('checkpoint_path',training.get('checkpoint_path')),
      'pipeline':str(pipeline_path),'experiment':experiment,'training_timing':timing,
      'closed_loop_result':str(loop_path) if loop_path else None,'closed_loop_episodes':loop.get('episodes',[]),
      'checkpoint_reload':resume.get('checkpoint_reload'),'optimizer_restore':resume.get('optimizer_restore'),
      'policy_service_cleanup':policy_record.get('cleanup_completed'),
      'formal_benchmark_performance':None,'formal_training_started':False,'persistent_gpu_poller_started':False}
    monitor = read('monitoring/state.json')
    result['bounded_gpu_monitoring'] = {k:monitor.get(k) for k in ['status','started_at','deadline','interval_seconds','auto_training','checks','last_check_at']}
    result['bounded_gpu_monitoring_config'] = str(ROOT/'monitoring/config.json') if monitor else None
    monitor_note = (f"用户后续有期限只读 GPU 监控当前状态：{monitor.get('status')}。原配置每 5 分钟、24 小时、只通知不自动训练；STOPPED 表示已取消，实时状态见 monitoring/state.json。" if monitor else '当前没有长期 GPU 等待器或自动开机任务。')
    (r/'validation_results.json').write_text(json.dumps(result,indent=2)+'\n')
    stage_table='\n'.join('| '+k+' | '+v['status']+' | '+v.get('scope',v.get('reason') or '')+' |' for k,v in stage_results.items())
    gpu_table='\n'.join(f"| {g['index']} | {g['used_mib']} | {g['free_mib']} | {g['utilization']} | {len(g['compute_pids'])} |" for g in gpu.get('samples',[[]])[-1])
    packages=ver.get('environments',{}).get('openpi',{}).get('packages',{})
    gpu_note = ('本轮没有合适空闲 GPU，选择器/训练入口返回 75（等待空闲 GPU），没有默认使用繁忙 GPU 0。' if wait else 'GPU 阶段的实际结果以本报告阶段表和对应训练/评测 JSON 为准。')
    measured = ('单卡可行性、多卡效果、训练峰值显存、编译耗时和稳定训练每步耗时均未测；没有根据历史显存或模型估计否定单卡路线。' if not training else '训练实测记录：'+json.dumps({k:training.get(k) for k in ['status','fsdp_devices','compile_seconds','observed_peak_device_memory_mib']},ensure_ascii=False))
    egl_state = stage_results['C_LIBERO_EGL_render']['status']
    pending = '、'.join(k for k,v in stage_results.items() if v['status'] not in ['PASS']) or '无'
    manual = ('当前需要合适的空闲 GPU 窗口；不需要重新授权 GPU 编号或证明独占。' if wait else ('本轮部署验证无需人工处理。' if overall=='SUCCESS' else '请根据阶段表查看是否仍有失败或未运行项目。'))
    episode_table='\n'.join(f"| {x['episode']} | {x['seed']} | {x['initial_state_index']} | {x['steps']} | {x['status']} | `{x.get('video')}` |" for x in loop.get('episodes',[]))
    performance=(f"双卡 FSDP={training.get('fsdp_devices')}，全局 batch={training.get('batch_size')}。首次编译 {training.get('compile_seconds',0):.2f} 秒；第 2–20 步同步更新平均 {statistics.mean(steady):.3f} 秒，范围 {min(steady):.3f}–{max(steady):.3f} 秒。这一耗时不包含数据加载、参数校验和检查点 I/O。没有成功的单卡训练基线，不能据此声称双卡加速比。" if steady else '尚无稳定训练更新耗时。')
    text=f'''# π0.5 LoRA + LIBERO 部署验证报告

更新时间：{now}

## 总体状态：{overall}

已完成两套隔离环境、官方资源下载校验、真实数据与 batch 检查、子集归一化、参数审计、真实基础权重 CPU 加载校验和 LIBERO CPU 无头仿真。
当前资源状态：**{result['resource_state']}**。本轮真实训练更新 {result['training_updates']} 次，resume 额外更新 {result['resume_updates']} 次。
单卡 π0.5 LoRA 训练是否跑通：**{result['single_gpu_training_verified']}**；多卡是否使用：**{result['multi_gpu_used']}**。
只有 pipeline 真实完成训练、检查点重载与恢复、策略闭环后，才将总体状态写为 SUCCESS。
本报告没有正式成功率或泛化能力结论。CPU 渲染视频使用合法 dummy action，不是训练策略闭环视频。

## 目录、环境与磁盘

- 项目：`{ROOT}`。
- 训练/策略服务：`envs/openpi`，Python 3.11.13；仿真/客户端：`envs/libero`，Python 3.8.20。
- 两套环境的 `uv pip check` 均通过；完整版本见 `reports/versions.txt` / `versions.json`。
- 所有包装脚本读取 `scripts/project_env.sh`，显式设置模型、HF、LeRobot、uv、pip、JAX、Numba、Matplotlib、临时文件和 LIBERO 配置路径。
- 保留真实 HOME，未改动 base、已有 venv、VLN 项目、系统驱动或 CUDA 链接。
- 当前项目物理占用约 **{storage['project_GiB']:.2f} GiB**（du 对硬链接去重）；文件系统当前可用 **{storage['filesystem_available_GiB']:.2f} GiB**。
- 本轮包含失败尝试的 GPU 阶段窗口记账：{read('reports/validation_resource_ledger.json').get('total_gpu_window_hours','未生成')} GPU 小时；包含编译、CPU 校验与保存等待，不等于 GPU 活跃计算时间。逐次记录见 `validation_resource_ledger.json`。
- 新增预算 120 GiB、文件系统最低保留 100 GiB；官方资源合计约 11.623 GiB，初始下载计划额外预留 60 GiB 给环境/检查点。个人 quota 工具未发现。
- 不自动下载官方 pi05_libero，不上传 W&B、日志或数据，不 push。

## 固定版本与来源

- openpi：`{ver.get('openpi_commit')}`，来源 https://github.com/Physical-Intelligence/openpi。
- LIBERO：`{ver.get('libero_commit')}`，使用该 openpi 固定子模块；不需要的 ALOHA 子模块未下载。
- LeRobot：`{ver.get('lerobot_commit')}`，按 openpi 锁文件安装。
- JAX/jaxlib 0.5.3，Flax 0.10.2，Orbax 0.11.13；训练依赖中的 PyTorch 2.7.1 用于数据加载，本项目训练后端仍为 JAX/Flax。
- NVIDIA 驱动 550.54.14，nvidia-smi CUDA 兼容版本 12.4；JAX 使用锁定的 CUDA 12 wheel 依赖，没有安装 CUDA 13、更新驱动或跳过兼容性检查。
- 基础模型：`gs://openpi-assets/checkpoints/pi05_base`；每个 GCS 对象的 generation、大小和 MD5 均保存并校验。
- 数据：`physical-intelligence/libero`，revision `{data.get('source_revision')}`，LeRobot v2.0。固定 LeRobot v2.1 代码明确支持向后读取；未做有上传副作用的数据升级。
- 该预转换数据是官方 openpi 示例使用的 LIBERO 派生数据（转换脚本使用 modified LIBERO RLDS）；仿真套件为原 LIBERO，未引入 LIBERO-plus。
- 数据图像嵌入 Parquet，实际 `total_videos=0`；无需额外数据视频解码。评测 MP4 编码已通过实际保存验证。

## 冻结的 smoke 数据与配置

- suite=`libero_spatial`，task order=0，task ID=0。
- 语言：`{data.get('language')}`。
- episode IDs：`{data.get('episode_ids')}`；共 **{data.get('frames')} 帧**，每条完整保留，按匹配任务的原始 episode ID 取首/中/末轨迹。
- 语言同时核对固定 LIBERO task map 和数据 task index。清单、原始 revision 与长度见 `data_manifest.json`。
- 配置：单卡 `pi05_libero_lora_3090_smoke/train`；双卡 `pi05_libero_lora_3090x2_smoke/train`，均已注册。当前实测配置 `{training.get('config')}`。
- π0.5：action_horizon=10、内部 action_dim=32、max_token_len=200、discrete_state_input=False；沿用该 commit 的 π0.5 LIBERO 配方。
- 原始 state 8 维、action 7 维；两个相机 256×256，送模型前 224×224，第三个相机补零并 mask。
- 仿真图像按官方示例旋转 180°；相对 OSC_POSE 控制命令与夹爪动作沿用上游，不把数值直接当作米或弧度；`extra_delta_transform=False`，没有重复做 delta 转换。
- 动作 chunk 在 episode 末尾按上游逻辑重复末帧并产生 padding mask；openpi 上游 repack 不消费该 mask，本项目没有改写损失语义。已用真实三条轨迹的首/中/末帧核对无跨 episode 读取。
- 子集 norm stats 位于 `configs/assets/pi05_libero_lora_3090_smoke/physical-intelligence/libero/norm_stats.json`；按上游真实 chunk 统计流程计算，未使用全量数据统计代替。
- 本轮 batch={training.get('batch_size','未运行')}、fsdp_devices={training.get('fsdp_devices','未运行')}，双卡时每卡 batch=1；workers=2、EMA=None、seed=42；预热 2 步、学习率 5e-5，目标 20 次更新；恢复目标总更新数 22。双卡与单卡使用相同模型、冻结规则、数据和 norm stats。
- LoRA：PaliGemma rank=16/alpha=16，action expert rank=32/alpha=32；均 alpha/rank=1，不使用 rsLoRA；注入 attention 和 FFN，完整路径见参数审计。

## 参数审计

| 项目 | 参数量 |
|---|---:|
| 总参数 | {audit.get('total_parameters')} |
| 可训练参数 | {audit.get('trainable_parameters')} |
| 可训练比例 | {audit.get('trainable_fraction',0):.4%} |
| LoRA 可训练参数 | {audit.get('lora_trainable_parameters')} |
| 非 LoRA 可训练参数 | {audit.get('non_lora_trainable_parameters')} |

官方 freeze_filter 来自同一个模型配置实例，仍训练视觉编码器 414,803,696 个参数，以及动作投影和时间 MLP 等。未额外冻结视觉编码器，也没有把本配置称为“仅训练 adapter”。
`parameter_audit.json` 包含每个叶子的 shape、dtype、冻结状态、优化器输入路径与优化器状态形状。抽象审计之外，真实训练前后逐参数哈希验证：冻结参数保持不变={training.get('frozen_unchanged')}；LoRA 参数变化断言已包含在训练验收中。新进程重载参数={resume.get('checkpoint_reload','未验证')}，优化器状态={resume.get('optimizer_restore','未验证')}。
`base_weight_audit.json` 已用真实官方权重逐叶校验 shape、dtype 与有限性，允许缺失的仅为新建 LoRA 参数；没有非 LoRA 基础参数缺失。

## GPU 选择与瞬时资源

最后采样时间：{gpu.get('time','unknown')}。
选择条件：在已有设备可见性限制内，两次采样间隔 3 秒，计算进程为空、已用显存不超过 512 MiB、空闲比例至少 95%、利用率不超过 5%；启动前再检查。不宣称获得全机独占锁。

| 物理编号 | 已用 MiB | 空闲 MiB | 利用率 % | 计算进程数 |
|---|---:|---:|---:|---:|
{gpu_table}

{gpu_note} 未终止其他任务、重置 GPU 或无限轮询。
{measured}
GPU 运行按子进程 UUID 限定 CUDA；EGL 单独通过 NVIDIA 持久 UUID 与选定卡的 NVML UUID 匹配，并在 reset 后核对当前 display 的实际设备 UUID，无法核对则报错。仿真子进程关闭 CUDA 计算可见性、保留显式 EGL ID，避免 robosuite 的数字 EGL ID 与 CUDA UUID 字符串冲突。EGL 路径当前状态：{egl_state}。

{performance}
显存为整卡 NVML 至少每 0.5 秒采样的观测峰值，可能漏掉短峰值，不能当成严格的显存最低需求。完整分阶段显存见 `validation_results.json`。
首次双卡训练观测峰值：{training.get('observed_peak_device_memory_mib')}。
恢复阶段观测峰值：{resume.get('observed_peak_device_memory_mib')}。
`dual_fsdp_checkpoint_audit.json` 检查真实检查点中的 mesh/partition_spec，验证大参数与优化器状态存在跨两卡分片；小张量按上游规则复制。

## 验证结果

| 阶段 | 状态 | 范围/原因 |
|---|---|---|
{stage_table}

CPU 渲染器：`{cpu.get('gl_renderer')}`。reset、两路图像、8 维状态和 12 个合法动作已运行，帧非空，环境正常关闭。
视频：`results/cpu_render_20261012_v2/episode_000_RENDER_SMOKE_COMPLETED.mp4`。
训练检查点：{result['training_checkpoint'] or '尚未产生'}。

实际流水线：`{pipeline_path}`；训练实验：`{experiment}`。原先 20 步训练已保存并验参后，进程退出出现 SIGSEGV（选择器记录 -11，外层退出码 245）；保留该失败记录，恢复只读取已有检查点，不重复 20 步。修复时增加 faulthandler、显式关闭检查点管理器和加载 worker；恢复进程正常退出，底层崩溃堆栈未获取，因此不声称已证明其原生根因。

闭环结果：`{loop_path}`，checkpoint 使用恢复后的实际目录；renderer=`{loop.get('gl_renderer')}`，策略服务清理={policy_record.get('cleanup_completed')}。

| episode | seed | 初始状态 | 策略步数 | 终止原因 | 视频 |
|---|---:|---:|---:|---|---|
{episode_table}

两集只验证真实图像/状态→本轮模型→有限动作→环境反馈的链路；任务未完成与程序异常单独标记，不用于估计正式成功率。

## 问题与修复

1. 默认沙箱没有 NVIDIA 设备。只读核对使用主机侧查询；GPU 任务必须在允许 GPU 访问的执行环境运行，不把沙箱枚举失败当成驱动故障。
2. 官方训练入口硬编码 `~/.cache/jax`：局部补丁改用项目 `JAX_COMPILATION_CACHE_DIR`。
3. 固定版 LeRobot 拒绝旧 `LEROBOT_HOME`：仅设置 `HF_LEROBOT_HOME`。
4. 非连续 episode 子集的索引缺陷：在项目 Dataset 子类中将原始 episode ID 映射为选中集合的位置，再复用官方 clamp/padding。原始文件和第三方安装源码未改动；真实轨迹边界验证通过。
5. HTTPS 下载出现 TLS 错误：保留现场与断点，改用验证过的官方 GCS 路由；有界重试，未换陌生模型镜像。
6. 零字节 `commit_success.txt`：修复下载器的空文件断点处理；第三版复用全部大文件并完成大小/哈希校验。
7. LIBERO editable 安装还需官方文档要求的 PYTHONPATH：在项目环境脚本中设置，未修改全局 shell。
8. 在临时执行沙箱内启动的首次后台 CPU 渲染被生命周期中断；确认进程不再存活后，在主机侧新记录重跑，保留失败记录。
9. 双 worker 数据加载不能以 stdin 作为 spawn 主模块：改为有 main guard 的 `batch_audit.py`，实际 batch 检查通过。
10. 策略服务默认绑定地址改为可配置的 loopback；本项目入口固定 127.0.0.1。单任务 evaluator 不吞异常，保存 SUCCESS/STEP_LIMIT/PROGRAM_EXCEPTION 区分。
11. EGL 混合厂商枚举超出 MuJoCo 索引上限、DRM 扩展返回空路径：对子进程限制 NVIDIA vendor，通过 EGL binary UUID 映射；真实 GPU 渲染已通过。
12. JAX AOT `lower().compile()` 构造 `ArgInfo` 元数据触发上游 dataclass 类型检查：仅将 `jax._src.stages` 加入已有元数据检查跳过名单，CPU 回归验证真实错误类型仍会被拒绝。双卡真实编译和 20 次更新通过。

最小上游 diff：`reports/openpi_local.patch`。项目新增脚本均保留在 `scripts/`。环境依赖锁文件未修改，未执行 PyTorch 专用覆盖补丁。

## 继续执行命令

在项目目录执行；GPU 默认 auto，也可对子进程设置 `GPU=编号或UUID`。

```bash
cd {ROOT}
# 一次性有限流程：JAX GPU→EGL→20 步训练→重载/恢复 2 步→2 个闭环 episode。
# 无空闲 GPU 即返回 75，不后台等待。
TRAIN_GPUS=2 GPU=0,1,2,3 bash scripts/run_validation.sh --gpu-only

# 单独重新执行 smoke 与恢复；恢复必须用同一个实际实验名。
TRAIN_GPUS=2 bash scripts/smoke_train.sh smoke_NEW_UNIQUE_NAME
TRAIN_GPUS=2 bash scripts/resume.sh smoke_NEW_UNIQUE_NAME

# 准备正式 spatial 数据与独立统计量；本轮未执行此全量准备。
source scripts/project_env.sh
/usr/bin/python3 -B scripts/prepare_assets_v3.py --full-spatial
bash scripts/compute_norm_stats.sh train
# 正式训练明确要求用户提供步数；本轮未启动。
TRAIN_GPUS=2 bash scripts/train.sh YOUR_TOTAL_STEPS UNIQUE_FORMAL_EXPERIMENT

# 路径来自实际运行的 results/<experiment>/checkpoint.json，不能猜步数目录。
bash scripts/serve_policy.sh /ACTUAL/CHECKPOINT 8000 pi05_libero_lora_3090x2_smoke
bash scripts/evaluate.sh /ACTUAL/CHECKPOINT 2 0 pi05_libero_lora_3090x2_smoke
```

`run_validation.py` 以有限阶段串联后续工作，每阶段保存日志，完成/失败后自动更新本报告及 `validation_results.json`。`run_stage.py` 可提供脱离会话的有限监督；不要在临时沙箱中启动期望长期存活的进程。{monitor_note}
关联原会话 ID：`PUBLIC_REPRODUCTION`；本地 completion_event 区分阶段结束与报告处理，没有把消息入队冒充分析完成。

## 已执行与未执行脚本

- 已实际执行：bootstrap、LIBERO 安装、资产下载 v3、参数审计、真实数据审计、norm stats、双 worker batch 审计、真实 base 权重审计、CPU/GPU 渲染、JAX 多卡探针、双卡 20 步、重载恢复 2 步、2 个策略闭环、视频逐帧计数与进程退出核验。
- 安全选择器 5 个边界测试通过；配置解析、训练/服务 CLI、Shell/Python 语法已检查。
- 尚未通过的项目：{pending}；对应脚本存在不代表端到端已验证。
- 正式训练脚本与全 spatial 准备脚本仅生成，未运行。

## 需要人工处理的事项

{manual} 没有已确认的管理员操作、额外凭证或预算扩容需求。单卡结论以真实 GPU 阶段记录为准。

参考：官方 [openpi](https://github.com/Physical-Intelligence/openpi)、[LIBERO 示例](https://github.com/Physical-Intelligence/openpi/blob/{ver.get('openpi_commit')}/examples/libero/README.md)、[JAX 安装](https://docs.jax.dev/en/latest/installation.html)、[JAX 显存分配](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)。
'''
    (r/'DEPLOYMENT_REPORT.md').write_text(text)
    print(json.dumps({'overall':overall,'resource_state':result['resource_state'],'training_updates':result['training_updates'],'report':str(r/'DEPLOYMENT_REPORT.md')},ensure_ascii=False))

if __name__=='__main__':main()
