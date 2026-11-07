"""Lab sensor bring-up, copied from the previously used start_robot_feedback.py."""


import inspect


def check_messages(cameras_only=False):
    import json
    import math
    import threading
    import time

    import rospy
    from sensor_msgs.msg import Image, JointState

    rospy.init_node("baseline_check", anonymous=True, disable_signals=True, disable_rosout=True)
    samples = {}
    completed = threading.Event()
    lock = threading.Lock()
    topics = ["/camera_" + side + "/color/image_raw" for side in "flr"]
    if not cameras_only:
        topics += ["/puppet/joint_left", "/puppet/joint_right"]
    def callback(msg, topic):
        stamp = msg.header.stamp.to_sec()
        if not 0 <= time.time() - stamp < 2:
            return
        if isinstance(msg, JointState):
            if list(msg.name) != ["joint" + str(i) for i in range(7)] or len(msg.position) != 7 or not all(map(math.isfinite, msg.position)):
                return
            value = list(msg.position)
        else:
            if not (msg.width > 0 and msg.height > 0 and msg.step > 0 and len(msg.data) == msg.height * msg.step):
                return
            value = [msg.width, msg.height, msg.encoding]
        with lock:
            prev = samples.get(topic)
            count = 1 if prev is None else prev[0] + int(stamp > prev[1])
            samples[topic] = (count, stamp, value)
            if all(t in samples and samples[t][0] >= 2 and time.time()-samples[t][1] < 2 for t in topics):
                completed.set()
    subs = [rospy.Subscriber(t, JointState if t.startswith("/puppet/") else Image, callback,
                             callback_args=t, queue_size=1, buff_size=4*1024*1024) for t in topics]
    try:
        ok = completed.wait(48)
        with lock:
            print(json.dumps({"ok":ok,"samples":samples}), flush=True)
    finally:
        for s in subs:
            s.unregister()
        rospy.signal_shutdown("feedback check complete")
    if not ok:
        raise SystemExit(1)


def passive_joint_feedback():
    """Publish only fresh complete feedback frames from the documented PiPER IDs."""
    import contextlib
    import select
    import socket
    import struct
    import time
    import rospy
    from sensor_msgs.msg import JointState

    rospy.init_node("baseline_can_feedback", disable_rosout=True)
    with contextlib.ExitStack() as stack:
        ports, publishers, frames = {}, {}, {}
        for side in ("left","right"):
            rx = stack.enter_context(socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW))
            # Exact standard data ID. Including CAN_ERR_FLAG would select error frames.
            rx.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                          b"".join(struct.pack("=II", k, 0xC00007FF) for k in range(0x2a5, 0x2a9)))
            rx.bind(("can_"+side,))
            ports[rx] = side
            frames[side] = {}
            publishers[side] = rospy.Publisher("/puppet/joint_"+side, JointState, queue_size=1)
        while not rospy.is_shutdown():
            readable, _, _ = select.select(list(ports), [], [], 0.2)
            for rx in readable:
                data = rx.recv(16)
                ident, length = struct.unpack_from("=IB",data)
                if length != 8 or ident not in range(0x2a5,0x2a9):
                    continue
                side = ports[rx]
                cache = frames[side]
                now = time.monotonic()
                cache[ident] = (now, data[8:])
                if len(cache)!=4 or now - min(t for t,_ in cache.values()) > 0.04:
                    continue
                joint = [n*0.017444/1000 for k in range(0x2a5,0x2a8) for n in struct.unpack(">ii",cache[k][1])]
                opening = struct.unpack_from(">i",cache[0x2a8][1])[0] / 1000000
                msg = JointState()
                msg.header.stamp = rospy.Time.now()
                msg.name = ["joint"+str(i) for i in range(7)]
                msg.position = [*joint, opening]
                publishers[side].publish(msg)
                cache.clear()


def bringup(probe, passive):
    import contextlib
    import collections
    import fcntl
    import hashlib
    import json
    import os
    from pathlib import Path
    import pwd
    import select
    import shutil
    import signal
    import socket
    import struct
    import subprocess
    import sys
    import tempfile
    import time
    import xml.etree.ElementTree as ET
    import xmlrpc.client

    def event(stage, **data):
        print("BASIC_EVENT " + json.dumps({"stage":stage, **data}), flush=True)

    home = Path("/home/agilex")
    if socket.gethostname().split(".")[0] != "agilex" or pwd.getpwuid(os.getuid()).pw_name != "agilex":
        raise SystemExit("Unexpected robot or account.")
    directory_lock = os.open(home, os.O_RDONLY | os.O_DIRECTORY)
    fcntl.flock(directory_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ros = Path("/opt/ros/noetic")
    arm = home / "cobot_magic/Piper_ros_private-ros-noetic"
    cam = home / "cobot_magic/camera_ws"
    sdk = home / "piper_sdk-0.3.2"
    doc = arm / "src/piper/launch/start_ms_piper.launch"
    driver = arm / "src/piper/scripts/piper_start_ms_node.py"
    camera_launch = cam / "src/ros_astra_camera/launch/multi_camera.launch"
    pins = {doc:"ef5b891b00e3266403211fcdde47e5b4b9577d764b7ed165aa09a7a005c9a7e6",
            driver:"228ad211d1eec7770c518a5c83a009fb50847851006e2209df4ef850dcd74fd6",
            camera_launch:"38ee8e258588cd72c2c04620400278620b0f961c92ddfeae343479df6c765710",
            camera_launch.with_name("dabai.launch"):"484d10f142f55a5c798a9f9e14e931891574ead4abf639228aa9b7a423f8898a",
            sdk/"piper_sdk/__init__.py":"d9c6912d275f552c9e68f15c67436d5f853cb339949a6a033a213de4d5b81e32",
            sdk/"piper_sdk/interface/piper_interface.py":"794b19c75bbdf305ba4ba892ffa31c01a9ffce315d07752e1631a3771a101e36"}
    for path, digest in pins.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Changed source: " + str(path))

    def ros_env(ws):
        clean = {"HOME":str(home), "USER":"agilex", "LOGNAME":"agilex", "PATH":"/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
                 "LANG":"C.UTF-8", "TERM":"dumb"}
        cmd = ['source "$1" >/dev/null && source "$2" >/dev/null && env -0',
               "_", str(ros/"setup.bash"), str(ws/"devel/setup.bash")]
        raw = subprocess.check_output(["/bin/bash","--noprofile","--norc","-c",*cmd], env=clean, timeout=12)
        env = dict(os.fsdecode(x).split("=",1) for x in raw.split(b"\0") if b"=" in x)
        env.update(ROS_MASTER_URI="http://127.0.0.1:11311", ROS_HOSTNAME="localhost", ROS_IP="127.0.0.1",
                   PYTHONDONTWRITEBYTECODE="1")
        return env

    event("checking local ROS workspaces")
    ae, ce = ros_env(arm), ros_env(cam)
    ae["PYTHONPATH"] = str(sdk) + os.pathsep + ae.get("PYTHONPATH","")
    pkg = subprocess.check_output([str(ros/"bin/rospack"),"find","astra_camera"],env=ce,text=True,timeout=10).strip()
    if Path(pkg).resolve() != camera_launch.parents[1].resolve():
        raise RuntimeError("Camera workspace resolves a different astra_camera: " + pkg)
    expected = {"CC1365304H8","CC1365302D0","CC1365304GC"}
    attached = set()
    for port in Path("/sys/bus/usb/devices").iterdir():
        try:
            if (port/"idVendor").read_text().strip() == "2bc5":
                attached.add((port/"serial").read_text().strip())
        except OSError:
            pass
    if not expected.issubset(attached):
        raise RuntimeError("Missing configured Orbbec serial(s): " + str(expected-attached))

    socket.setdefaulttimeout(2)
    master = xmlrpc.client.ServerProxy("http://127.0.0.1:11311")
    def graph():
        code, _, value = master.getSystemState("/baseline_setup")
        if code != 1:
            raise RuntimeError("Could not inspect ROS master.")
        return value
    try:
        state = graph()
        existing_master = True
        nodes = {node for section in state for _, names in section for node in names}
        if nodes - {"/rosout"}:
            raise RuntimeError("Existing ROS nodes, no duplicate launch: " + ", ".join(sorted(nodes)))
        if master.getParam("/baseline_setup","/use_sim_time")[2] is True:
            raise RuntimeError("Existing master uses simulation time.")
    except ConnectionRefusedError:
        existing_master = False

    ip = shutil.which("ip", path="/usr/sbin:/sbin:/usr/bin:/bin")
    mapping = {"can_left":"1-13:1.0","can_right":"1-12:1.0"}
    def links():
        return {d.name:(d/"device").resolve().name for d in Path("/sys/class/net").iterdir()
                if (d/"type").read_text().strip()=="280"}
    def info(name):
        return json.loads(subprocess.check_output([ip,"-j","-details","link","show","dev",name],text=True))[0]
    def up(record):
        return "UP" in record["flags"]
    plan = []
    found = links()
    for name, usb in mapping.items():
        matched = [n for n,v in found.items() if v == usb]
        if len(matched) != 1 or ((Path("/sys/class/net") / name).exists() and matched[0] != name):
            raise RuntimeError("Ambiguous interface mapping for " + name)
        old = matched[0]
        before = info(old)
        bt = before["linkinfo"]["info_data"].get("bittiming",{})
        if up(before) and (old != name or bt.get("bitrate") != 1000000):
            raise RuntimeError("An active arm link would have to be reconfigured: " + old)
        plan.append((name, old, before))
    arm_names = {old for _,old,_ in plan}
    for table in Path("/proc/net/can").glob("rcvlist_*"):
        for line in table.read_text().splitlines():
            fields = line.split()
            if len(fields) >= 5 and fields[0] in arm_names | {"any"}:
                raise RuntimeError("Another CAN reader is attached to " + fields[0] + ". Existing owner left in place.")
    event("verified USB mapping", arms={n:o for n,o,_ in plan}, cameras=sorted(expected))
    if any(not up(record) for _,_,record in plan):
        print("Linux sudo may ask for the agilex password to configure the two arm CAN interfaces.", flush=True)
        subprocess.run(["/usr/bin/sudo","-v"],check=True,timeout=180)

    def change(*args):
        subprocess.run(["/usr/bin/sudo","-n",ip,"link","set","dev",*args],check=True,timeout=8)

    children = []
    touched = []
    def launch(label, argv, env, folder):
        logfile = Path(folder) / (label+".log")
        with logfile.open("wb") as out:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=out,
                                     cwd=folder, env=env, start_new_session=True)
        children.append((label,child,logfile))
    def running():
        for label,child,logfile in children:
            if child.poll() is not None:
                raise RuntimeError(label+" exited: "+logfile.read_bytes()[-1800:].decode(errors="replace"))
    def shutdown_child(p):
        if p.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(p.pid,signal.SIGINT)
            try:
                p.wait(6)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(p.pid,signal.SIGTERM)
                p.wait(5)
    def stop_requested(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig,stop_requested)
    ready = False
    with tempfile.TemporaryDirectory(prefix="agilex-feedback-") as folder:
        try:
            if not existing_master:
                launch("roscore",[str(ros/"bin/roscore"),"-p","11311"],ce,folder)
                for i in range(36):
                    running()
                    try:
                        graph()
                        break
                    except ConnectionRefusedError:
                        time.sleep(0.3)
                else:
                    raise RuntimeError("ROS master did not start.")
            launch("cameras",[str(ros/"bin/roslaunch"),"--wait",str(camera_launch)],ce,folder)
            for name,old,record in plan:
                if up(record):
                    continue
                if links().get(old) != mapping[name] or up(info(old)):
                    raise RuntimeError("Link state changed before configuration: " + old)
                touched.append((name,old,record))
                if name != old:
                    change(old,"name",name)
                change(name,"type","can","bitrate","1000000")
                change(name,"up")
                if not up(info(name)) or links().get(name) != mapping[name]:
                    raise RuntimeError("CAN activation failed: " + name)
            # Check for sensor frames on the wires before trusting a driver's default zero arrays.
            ids = {0x2a5,0x2a6,0x2a7,0x2a8}
            counts = {name:{k:0 for k in ids} for name in mapping}
            observed_ids = {name:collections.Counter() for name in mapping}
            def bus_stats():
                result = {}
                for name in mapping:
                    rec = info(name)
                    d = rec["linkinfo"]["info_data"]
                    sysfs = Path("/sys/class/net")/name/"statistics"
                    result[name] = {"state":d.get("state"),"bitrate":d.get("bittiming",{}).get("bitrate"),
                                    **{k:int((sysfs/k).read_text()) for k in ("rx_packets","rx_errors","rx_dropped","tx_packets","tx_errors")}}
                return result
            before_rx = bus_stats()
            with contextlib.ExitStack() as stack:
                sockets = {}
                for name in mapping:
                    rx = stack.enter_context(socket.socket(socket.AF_CAN,socket.SOCK_RAW,socket.CAN_RAW))
                    # Receive ordinary bus traffic so an alternate feedback-ID range is observable.
                    rx.setblocking(False)
                    rx.bind((name,))
                    sockets[rx] = name
                end = time.monotonic()+12
                while time.monotonic()<end and min(min(v.values()) for v in counts.values())<2:
                    readable,_,_ = select.select(list(sockets),[],[],0.25)
                    for rx in readable:
                        packet = rx.recv(16)
                        ident, length = struct.unpack_from("=IB",packet)
                        observed_ids[sockets[rx]][ident] += 1
                        if ident in ids and length == 8:
                            counts[sockets[rx]][ident] += 1
            event("CAN feedback", counts=counts, before=before_rx, after=bus_stats(),
                  observed={k:{hex(i):n for i,n in c.most_common(64)} for k,c in observed_ids.items()})
            arm_feedback = all(min(v.values()) >= 2 for v in counts.values())
            source = "none"
            if not arm_feedback:
                event("arm driver skipped", reason="No complete hardware feedback on both arms. No automatic retry.")
            # Append an existing python-can dependency after Noetic/system paths. No environment installation.
            sites = []
            for dist in ("miniconda3","anaconda3","miniforge3"):
                base = home/dist
                sites.extend(sorted(base.glob("envs/*/lib/python*/site-packages")))
                sites.extend(sorted(base.glob("lib/python*/site-packages")))
            sites.extend(sorted((home/".local/lib").glob("python*/site-packages")))
            sites.extend(sorted((home/"xmh/openpi-agilex/.venv/lib").glob("python*/site-packages")))
            py_minor = "python%d.%d"%sys.version_info[:2]
            candidates = sorted((s for s in sites if (s/"can/__init__.py").is_file()),
                                key=lambda s:(s.parent.name!=py_minor,"aloha" not in s.parts,str(s)))
            inspect_can = subprocess.run(["/usr/bin/python3","-B","-c",
                "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('can') else 1)"],env=ae)
            site = candidates[0] if candidates and inspect_can.returncode else None
            original = arm_feedback and (site is not None or inspect_can.returncode == 0)
            if site:
                event("using existing python-can", path=str(site/"can"))
            if original:
                source = "site driver, mode 0"
            # sys.path.append keeps system ROS, NumPy and tf ahead of the other environment.
            prefix = ["-c","import runpy,sys;site,script=sys.argv[1:3];sys.path.append(site);sys.argv=sys.argv[2:];runpy.run_path(script,run_name='__main__')",str(site)] if site else []
            arm_start = len(children)
            for node in (ET.parse(doc).getroot().findall("node") if original else []):
                port = node.find("param[@name='can_port']").attrib["value"]
                side = {"can_left":"left", "can_right":"right"}[port]
                remaps = {r.attrib["from"]:r.attrib["to"] for r in node.findall("remap")}
                # The stock feedback mode also emits master-arm telemetry on its command topic.
                # Put that telemetry on an observation-only name for this session.
                remaps["/master/joint_states"] = "/baseline_feedback/master_joint_"+side
                launch("piper_"+side,["/usr/bin/python3","-B",*prefix,str(driver),"__name:=baseline_piper_"+side,
                        "_can_port:=can_"+side,"_mode:=0","_auto_enable:=false",
                        *[k+":="+v for k,v in remaps.items()]],ae,folder)
            if original:
                for i in range(35):
                    if any(p.poll() is not None for _,p,_ in children[arm_start:]):
                        original = False
                        break
                    pubs = dict(graph()[0])
                    if all("/puppet/joint_"+side in pubs for side in ("left","right")):
                        break
                    time.sleep(0.3)
                else:
                    original = False
                if not original:
                    for label,p,logfile in children[arm_start:]:
                        shutdown_child(p)
                        event("site driver startup diagnostic",component=label,
                              message=logfile.read_bytes()[-1000:].decode(errors="replace"))
                    del children[arm_start:]
            if arm_feedback and not original:
                source = "passive SocketCAN reader"
                event("using passive CAN feedback", reason="Site driver dependency or startup unavailable")
                launch("passive",["/usr/bin/python3","-B","-c",passive],ae,folder)
            event("checking live cameras and joint samples")
            verified = subprocess.run(["/usr/bin/python3","-B","-c",probe+"\ncheck_messages("+repr(not arm_feedback)+")"],env=ae,stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT,timeout=62,text=True)
            if verified.returncode:
                running()
                raise RuntimeError("Live message check: "+verified.stdout[-2000:])
            observed = next(json.loads(line) for line in reversed(verified.stdout.splitlines()) if line.startswith('{"ok":'))
            running()
            if not observed["ok"]:
                raise RuntimeError("Topic check incomplete.")
            ready = True
            event("READY" if arm_feedback else "CAMERAS_READY", samples=observed["samples"],
                  feedback_source=source, site_arm_driver_started=original, auto_enable=False)
            print("Keep this window open. Ctrl+C stops this session's ROS nodes.",flush=True)
            while True:
                running()
                time.sleep(1)
        except KeyboardInterrupt:
            event("session closed")
        except Exception as exc:
            event("failed", error=str(exc))
            for label,child,logfile in children:
                lines = logfile.read_bytes()[-6000:].decode(errors="replace").splitlines()
                hints = [l[:360] for l in lines if any(w in l.lower() for w in ("error","failed","exception","permission","no module","not found"))]
                if hints:
                    event("diagnostic", component=label, messages=hints[-4:])
            raise
        finally:
            for _,p,_ in reversed(children):
                shutdown_child(p)
            if not ready:
                for name,old,record in reversed(touched):
                    if links().get(name) == mapping[name]:
                        try:
                            change(name,"down")
                            old_bt = record["linkinfo"]["info_data"].get("bittiming",{}).get("bitrate")
                            if old_bt:
                                change(name,"type","can","bitrate",str(old_bt))
                            if old!=name:
                                change(name,"name",old)
                        except (OSError,subprocess.SubprocessError) as exc:
                            event("check CAN state", interface=name, error=str(exc))
            event("session end", was_ready=ready, can_now=links())


if __name__ == "__main__":
    bringup(inspect.getsource(check_messages),
            inspect.getsource(passive_joint_feedback) + "\npassive_joint_feedback()\n")
