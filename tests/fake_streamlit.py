"""Minimal recording stand-in for streamlit so UI code paths can be smoke-tested without the real package."""
import sys
import types


class _Ctx:
    def __init__(self, st): self._st = st
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name): return getattr(self._st, name)
    def __iter__(self): return iter(())


class _SessionState(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v


class FakeUpload:
    def __init__(self, name, data, mime=""):
        self.name, self._d, self.type = name, data, mime
    def getvalue(self): return self._d
    def getbuffer(self): return memoryview(self._d)
    def read(self): return self._d


class FakeSt(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.calls, self.session_state = [], _SessionState()
        self.sidebar = _Ctx(self)
        self.buttons_pressed, self.file_upload, self.camera = set(), None, None
        self.keyed_uploads, self.mic = {}, None
        self.slider_value = None

    def cache_resource(self, *a, **k):
        if a and callable(a[0]): return a[0]
        return lambda f: f
    cache_data = cache_resource

    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Ctx(self) for _ in range(n)]
    def tabs(self, labels): return [_Ctx(self) for _ in labels]
    def expander(self, *a, **k): self.calls.append(("expander", a[0] if a else "")); return _Ctx(self)
    def spinner(self, *a, **k): return _Ctx(self)
    def empty(self): return _Ctx(self)
    def progress(self, *a, **k): return _Ctx(self)
    def button(self, label, key=None, **k): return (key or label) in self.buttons_pressed
    def file_uploader(self, *a, key=None, **k): return self.keyed_uploads.get(key) if key else self.file_upload
    def audio_input(self, *a, **k): return self.mic
    def camera_input(self, *a, **k): return self.camera
    def slider(self, label, min_value=0, max_value=1, value=0, **k): return value if self.slider_value is None else self.slider_value
    def radio(self, label, options=None, index=0, **k): return options[index]
    def selectbox(self, label, options, index=0, **k): return list(options)[index]
    def rerun(self): pass
    def set_page_config(self, **k): pass
    def stop(self): raise SystemExit

    def __getattr__(self, name):
        if name.startswith("__"): raise AttributeError(name)
        def rec(*a, **k):
            self.calls.append((name, a[0] if a and isinstance(a[0], str) else ""))
            return _Ctx(self) if name in ("container", "status") else None
        return rec

    def texts(self, kind): return [c[1] for c in self.calls if c[0] == kind]


def install() -> FakeSt:
    fake = FakeSt()
    sys.modules["streamlit"] = fake
    return fake
