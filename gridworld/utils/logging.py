def log(s, file=None, disp=True, write=True, **kwargs):
    if disp:
        print(s, **kwargs)
    if write:
        assert file
        print(s, file=file, **kwargs)


def log_tabular(vals, file_csv, file_txt=None, keys=None, formats=None, **kwargs):
    log(','.join([str(x) for x in vals]), file=file_csv, disp=False, **kwargs)
    if formats is not None:
        assert len(formats) == len(vals)
        vals = [x[0] % x[1] for x in zip(formats, vals)]
    if keys is not None:
        assert len(keys) == len(vals)
        log(' | '.join(['%s: %s' % (x[0], str(x[1])) for x in zip(keys, vals)]), file=file_txt, **kwargs)


def clear_file(path):
    open(path, 'w').close()
