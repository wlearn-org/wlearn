"""Optional dimensionality reduction, implemented by wlearn-dimred."""
try:
    from wlearn_dimred import PCA, TSNE, UMAP, TriMap
except ModuleNotFoundError as exc:
    if exc.name != 'wlearn_dimred':
        raise
    raise ImportError('Install wlearn[dimred] to use wlearn.dimred') from exc

__all__ = ['PCA', 'TSNE', 'UMAP', 'TriMap']
