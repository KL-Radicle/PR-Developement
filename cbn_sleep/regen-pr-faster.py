import json

from ParticipantReport import get_hash
from templates.templating_functions import generate_from_document

study_values = {

    'primary_outcome': 'sleep',
    'comparison_group': 'the control group',
    'study_size': '2738',
    'top_states': 'CA and TX',
    'top_demographic': 'Females under 40'
}


def generate_pr(args):
    code = get_hash(args['email'])
    args = args | study_values
    html = generate_from_document(args,
                                  doc='C:/Users/kaus/PycharmProjects/Data-Insights/Participant Report/templates/email_template.html',
                                  savepath=f'out/{code}.html',
                                  study_name_s3='cbn_sleep',
                                  )
    return code, html


filter = json.load(open('filter.json', 'r'))
# for args in tqdm(filter.values()):
#     generate_pr(args)
# break
# for args in tqdm(sub_formaters.values()):
#     generate_pr(args)
res = list(map(generate_pr, filter.values()))
