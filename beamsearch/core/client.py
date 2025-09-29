from qisabelle.client.session import QIsabelleServerError
from qisabelle.client.session import QIsabelleSession as _QIsabelleSession


class QIsabelleSession(_QIsabelleSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_state_name = "init"


__all__ = ["QIsabelleServerError", "QIsabelleSession"]
