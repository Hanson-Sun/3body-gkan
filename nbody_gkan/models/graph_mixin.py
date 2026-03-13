class GraphMixin:
    def summary(self):
        total = sum(p.numel() for p in self.parameters())
        print("=" * 50)
        print(f"  {self.__class__.__name__}")
        print("=" * 50)
        print(f"  Parameters:  {total:,}")
        print(f"  Input dim:   {getattr(self, 'n_f',      'N/A')}")
        print(f"  Message dim: {getattr(self, 'msg_dim',  'N/A')}")
        print(f"  Hidden dim:  {getattr(self, 'hidden',   'N/A')}")
        print(f"  Output dim:  {getattr(self, 'ndim',     'N/A')}")
        print("=" * 50)