_variables = ['momenta']

def _init(store):
    store.create_variable(
        'momenta', 'numpy.float32', dimensions=('ndim',),
        description="the momentum associated with the given degree of freedom",
        chunksizes=(1, 'ndim')
    )
