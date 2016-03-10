"""
Attributes
----------
momenta : numpy.ndarray, shape=(ndim,), dtype=numpy.float32
    momenta of the system
"""


attributes = ['momenta']
numpy = ['momenta']

def netcdfplus_init(store):
    store.create_variable(
        'momenta', 'numpy.float32', 
        dimensions=('ndim',),
        description="the momentum associated with the given degree of freedom",
        chunksizes=(1, 'ndim')
    )
