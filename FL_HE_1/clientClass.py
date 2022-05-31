import phe
class Client:
    """Runs linear regression with local data or by gradient steps,
    where gradient can be passed in.

    Using public key can encrypt locally computed gradients.
    """

    def __init__(self, name, X, y, cat_feat):
        self.name = name
        self.pubkey = None
        self.X, self.y, self.cat_feat = X, y, cat_feat
    
    def set_pubkey(self, pubkey):
        self.pubkey = pubkey
        