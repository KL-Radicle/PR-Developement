from os import listdir
from os.path import isfile, join

from icecream import ic
from tqdm import tqdm

report = 'dstress'
base_path = f'./{report}/out/'

files = [f"{base_path}{f}" for f in listdir(base_path) if isfile(join(base_path, f))]


def filter_replace_string(f_in, f_out, args):
    # ic(args, f_in, f_out)
    replace_args = args['to_replace'], args['target']
    # ic(replace_args)
    # return
    replaced = open(f_in, 'r').read().replace(*replace_args)
    f_out = open(f_out, 'w')
    f_out.write(replaced)
    return True


def apply_filter(fitler_function, file_list, suffix='', rename=False, args=None):
    if args is None:
        args = {}
    if rename:
        assert len(rename) == len(file_list)
        out_names = rename
    else:
        out_names = file_list
    if suffix != '':
        out_names = [f"{'.'.join(x.split('.')[:-1])}{suffix}.{x.split('.')[-1]}" for x in out_names]

    result = []
    for fi, fo in tqdm(zip(file_list, out_names)):
        result.append(fitler_function(fi, fo, args))

    return result


replace_map = {
    'to_replace': 'https://9251305.fs1.hubspotusercontent-na1.net/hub/9251305/hubfs/Screen%20Shot%202022-05-27%20at%202.26.22%20PM.png?width=520&amp;upscale=true&amp;name=Screen%20Shot%202022-05-27%20at%202.26.22%20PM.png',
    'target': 'https://radicle-production.s3.us-west-2.amazonaws.com/dstress/map.png'
}

replace_logo = {
    'to_replace' :
"""<a href="https://www.radiclescience.com"> 
                                                <img alt="Logo"
                                                     src="https://9251305.fs1.hubspotusercontent-na1.net/hub/9251305/hubfs/Screen%20Shot%202022-05-24%20at%202.29.08%20PM.png?width=1200&amp;upscale=true&amp;name=Screen%20Shot%202022-05-24%20at%202.29.08%20PM.png"
                                                     style="outline:none; text-decoration:none; -ms-interpolation-mode:bicubic; max-width:100%; font-size:16px"
                                                     width="600" align="middle>
    </a>
""",
    'target' :
        """<a href="https://www.radiclescience.com"> 
                                                <img alt="Logo"
                                                     src="https://9251305.fs1.hubspotusercontent-na1.net/hub/9251305/hubfs/Screen%20Shot%202022-05-24%20at%202.29.08%20PM.png?width=1200&amp;upscale=true&amp;name=Screen%20Shot%202022-05-24%20at%202.29.08%20PM.png"
                                                     style="outline:none; text-decoration:none; -ms-interpolation-mode:bicubic; max-width:100%; font-size:16px
                                                     width=600 align=middle">
													 
    </a>
"""
}

apply_filter(filter_replace_string, files, suffix='', args=replace_logo)
# filter_replace_string([files[0]], [f'{files[0]}_test.html'], args=replace_map)
