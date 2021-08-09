# (helper) parse_requirements
def parse_requirements(srcpath, comment_char='#'):
    with open(srcpath, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    reqs = []
    for l in lines:
        # filer all comments
        if comment_char in l:
            l = l[:l.index(comment_char)].strip()
        # skip directly installed dependencies
        if l.startswith('http'):
            continue
        if l:  # if requirement is not empty
            reqs.append(l)
    return reqs