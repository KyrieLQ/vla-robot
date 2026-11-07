"""Resolve NVIDIA EGL devices by persistent UUID, independently of CUDA ordinals."""
import ctypes, json, os, pathlib, uuid

def _binary_uuid(lib, device):
    ptr=lib.eglGetProcAddress(b'eglQueryDeviceBinaryEXT')
    if not ptr:raise RuntimeError('EGL_EXT_device_persistent_id unavailable')
    query=ctypes.CFUNCTYPE(ctypes.c_uint,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,
                          ctypes.c_void_p,ctypes.POINTER(ctypes.c_int))(ptr)
    value=(ctypes.c_ubyte*16)();size=ctypes.c_int()
    if not query(device,0x335C,16,value,ctypes.byref(size)) or size.value!=16:
        raise RuntimeError('Cannot read EGL device UUID')
    return 'GPU-'+str(uuid.UUID(bytes=bytes(value)))

def verify_current_egl(expected_uuid):
    lib=ctypes.CDLL('libEGL.so.1');lib.eglGetProcAddress.argtypes=[ctypes.c_char_p];lib.eglGetProcAddress.restype=ctypes.c_void_p
    lib.eglGetCurrentDisplay.restype=ctypes.c_void_p
    display=lib.eglGetCurrentDisplay()
    if not display:raise RuntimeError('No current EGL display to verify')
    ptr=lib.eglGetProcAddress(b'eglQueryDisplayAttribEXT')
    if not ptr:raise RuntimeError('EGL display-device query unavailable')
    query=ctypes.CFUNCTYPE(ctypes.c_uint,ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(ctypes.c_ssize_t))(ptr)
    device=ctypes.c_ssize_t()
    if not query(display,0x322C,ctypes.byref(device)):raise RuntimeError('EGL display-device query failed')
    actual=_binary_uuid(lib,ctypes.c_void_p(device.value))
    if actual.lower()!=expected_uuid.lower():raise RuntimeError('Actual EGL render device differs from selected GPU')
    return actual

def select_egl():
    # Keep the mapper and MuJoCo on the same NVIDIA EGL vendor. Mixed Mesa/NVIDIA
    # enumeration on this host exceeds MuJoCo 3.2.3's default ten-device buffer.
    vendor=pathlib.Path('/usr/share/glvnd/egl_vendor.d/10_nvidia.json')
    if not vendor.is_file():raise RuntimeError('NVIDIA EGL vendor manifest is unavailable')
    os.environ['__EGL_VENDOR_LIBRARY_FILENAMES']=str(vendor)
    selection=json.loads(pathlib.Path(os.environ['VLA_GPU_SELECTION']).read_text())['selected']
    if len(selection)!=1:raise RuntimeError('Simulation validation requires one selected physical GPU')
    wanted=selection[0]
    lib=ctypes.CDLL('libEGL.so.1');lib.eglGetProcAddress.argtypes=[ctypes.c_char_p];lib.eglGetProcAddress.restype=ctypes.c_void_p
    ptr=lib.eglGetProcAddress(b'eglQueryDevicesEXT')
    if not ptr:raise RuntimeError('EGL_EXT_device_enumeration unavailable')
    query=ctypes.CFUNCTYPE(ctypes.c_uint,ctypes.c_int,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int))(ptr)
    devices=(ctypes.c_void_p*32)();n=ctypes.c_int()
    if not query(32,devices,ctypes.byref(n)):raise RuntimeError('EGL enumeration failed')
    rows=[]
    for i in range(n.value):
        # Do not query an unavailable DRM path: that can leave a sticky EGL error.
        rows.append({'egl_index':i,'uuid':_binary_uuid(lib,devices[i])})
    matches=[r for r in rows if r['uuid'].lower()==wanted['uuid'].lower()]
    if len(matches)!=1:raise RuntimeError('Cannot unambiguously map selected GPU UUID to EGL: '+json.dumps(rows))
    if n.value>10:raise RuntimeError('NVIDIA EGL enumeration still exceeds pinned MuJoCo capacity')
    record={'selected_gpu':wanted,'egl_devices':rows,'matched_egl_index':matches[0]['egl_index'],
            'egl_vendor_manifest':str(vendor),'enumerated_device_count':n.value}
    os.environ['MUJOCO_EGL_DEVICE_ID']=str(matches[0]['egl_index']);os.environ['MUJOCO_GL']='egl';os.environ['PYOPENGL_PLATFORM']='egl'
    return record
