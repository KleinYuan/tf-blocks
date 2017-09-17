def deep_log(func):
    def wrapper(*args, **kwargs):
        space = '          '
        func_str = func.__name__
        args_str = str(args)
        kwargs_str = ', '.join([':'.join([str(j) for j in i]) for i in kwargs.iteritems()])
        # stout = cStringIO.StringIO()
        # stream = stout.getvalue()

        print '---------------------------------------'
        print 'Function Name : %s \n' \
              '%sArgs: %s \n' \
              '%sKwargs: %s \n' \
              % (func_str, space, args_str, space, kwargs_str)
        print '---------------------------------------'
        return func(*args, **kwargs)

    return wrapper
