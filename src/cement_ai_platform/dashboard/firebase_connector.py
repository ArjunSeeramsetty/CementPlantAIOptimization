from typing import Any, Dict


class FirebaseConnector:
    """Firebase connector with lazy initialization.

    Requires `firebase-admin` when used.
    """

    def __init__(self, service_account_path: str | None = None):
        self._db = None
        self._cred_path = service_account_path

    def _ensure_client(self):
        if self._db is not None:
            return
        try:
            import firebase_admin  # type: ignore
            from firebase_admin import credentials, firestore  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("firebase-admin not installed") from exc
        if not firebase_admin._apps:
            if not self._cred_path:
                raise RuntimeError("service_account_path is required to init Firebase")
            cred = credentials.Certificate(self._cred_path)
            firebase_admin.initialize_app(cred)
        self._db = firestore.client()

    def push_update(self, doc_path: str, payload: Dict[str, Any]) -> None:
        self._ensure_client()
        collection, _, doc = doc_path.partition("/")
        self._db.collection(collection).document(doc).set(payload, merge=True)



