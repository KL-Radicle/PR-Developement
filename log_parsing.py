import json

from icecream import ic

target_log = \
    'EMAIL_DISTRIBUTION-(18-10)-21_07_2022.log'
f = open(target_log)

parsed_log = {}
for i, l in enumerate(f.readlines()):
    try:
        tokens = l.split(':')
        type = tokens[0]
        caller = tokens[1]
        message = ''.join(tokens[2:])
        # ic(i, type, caller, message)
        parsed_log[i] = {
            'type' : type,
            'caller' : caller,
            'message' : message
            }
    except:
        # print('FAILED TO PARSE, ', l)
        ...

filtered_logs = {}

### FILTERING CODE HEER
for k, v in parsed_log.items():
    # ic(k, v)
    if v['caller'] == 'root' and 'sent' in v['message']:
        filtered_logs[k] = v
    # if 'error' in v['message']:
    #     filtered_logs[k] = v
# lines = []
# for k, v in filtered_logs:

ic(filtered_logs)
print(len(filtered_logs))
# json.dump(filtered_logs, open('obx_pr_output.json', 'w'))