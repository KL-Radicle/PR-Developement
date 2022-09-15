import dominate
from icecream import ic

from ParticipantReport import get_hash
from graphing_lib import patient_vs_cohort_line

# cache
cache = {}
cache['page_str'] = None


def generate_from_document(formatters, doc='./email_template.html', savepath=None, study_name_s3='default_PR_repo'):
    """
    Usage: generate_from_document(test_formatter, savepath='testout2.html')

    :param formatters:

    formatter_ex = {
        'first_name': 'TEST_NAME',
        'email': 'kaus@radiclescience.com',
        'conditions':
            {
                'Anxiety': {
                    'condition_valid': True,
                    'invert': True,  # means LOWER scores are BETTER
                    'scores_cohort': [4, 1, 2, 3],
                    'scores_individual': [6, 1, 2, 3]
                },

                'Wellbeing': {
                    'condition_valid': True,
                    'invert': False,  # means HIGHER scores are BETTER
                    'scores_cohort': [10, 11, 12, 13],
                    'scores_individual': [5, 3, None, 5]
                },

                'Sleep': {
                    'condition_valid': True,
                    'invert': True,  # means HIGHER scores are BETTER
                    'scores_cohort': [5, 7, 3, 8],
                    'scores_individual': [5, None, None, 4]
                },

                'Pain': {
                    'condition_valid': True,
                    'invert': True,  # means HIGHER scores are BETTER
                    'scores_cohort': [5, 7, 3, 8],
                    'scores_individual': [5, 2, 3, 4]
                },
            },

        'primary_outcome': 'Pain',
        'comparison_group': 'control',
        'study_size': '2323',
        'top_states': 'CA and TX',
        'top_demographic': 'Females over 40',
    }

    :param doc: template to generate from. Must define fields in @replacements and 'REPLACE_ME'.
    :param savepath:
    :return:
    """
    email = formatters['email']
    # put in graphs and condition name into a list
    sections = [
        (patient_vs_cohort_line(x['scores_individual'], x['scores_cohort'],
                                text_fmt={'invert': x['invert'], 'condition': k, 'xlabel': ''},
                                text_offset_y=-120, text_offset_x=200, text_size=24, marker_size_mult=0.75,
                                hide_legend=True, save_s3=f'{study_name_s3}/{get_hash(email)}/graph_{k}.png', height=600, width=1100
                                ),
         k)
        for (k, x) in formatters['conditions'].items() if x['condition_valid']
    ]

    sections = generate_section_code(sections)

    if cache['page_str'] is not None:
        page_str = cache['page_str']
    else:
        page_str = open(doc).read()
    # page = dominate.document(page_str)
    reconstructed = page_str.replace('REPLACE_ME', sections)
    replacements = [
        ('REPLACE_FIRST_NAME', formatters['first_name']),
        # ('Wellbeing', 'Well-being'),
        ('REPLACE_OUTCOME', formatters['primary_outcome']),  # pain, anxiety etc
        ('REPLACE_TOOK_WHAT', formatters['comparison_group']),  # nothing/those who took placebo/those who took X
        ('REPLACE_STUDY_SIZE', formatters['study_size']),  #
        ('REPLACE_TOP_STATES', formatters['top_states']),  #
        ('REPLACE_TOP_DEMOGRAPHIC', formatters['top_demographic']),  #
        ('https://9251305.fs1.hubspotusercontent-na1.net/hub/9251305/hubfs/Artboard%20214-4.png?width=1120&upscale=true&name=Artboard%20214-4.png', 'https://9251305.fs1.hubspotusercontent-na1.net/hub/9251305/hubfs/Artboard%20214-Jul-01-2022-03-56-08-11-AM.png?width=1120&upscale=true&name=Artboard%20214-Jul-01-2022-03-56-08-11-AM.png')
    ]
    for arg1, arg2 in replacements:
        reconstructed = reconstructed.replace(arg1, arg2)

    if savepath is not None:
        open(savepath, 'w').write(reconstructed)
    return reconstructed


def generate_section_code(sections):
    def generate_one_section_code(image_link, condition_name):
        images = {
            'Wellbeing': 'https://9251305.fs1.hubspotusercontent-na1.net/hub/9251305/hubfs/Artboard%20215-1.png?width=1120&upscale=true&name=Artboard%20215-1.png',
            'Sleep': 'https://9251305.fs1.hubspotusercontent-na1.net/hub/9251305/hubfs/Artboard%20215_1-1.png?width=1120&upscale=true&name=Artboard%20215_1-1.png',
            'Anxiety': 'https://9251305.fs1.hubspotusercontent-na1.net/hub/9251305/hubfs/Artboard%20215_2-1.png?width=1120&upscale=true&name=Artboard%20215_2-1.png',
            'Pain': 'https://9251305.fs1.hubspotusercontent-na1.net/hub/9251305/hubfs/Artboard%20215_5.png?width=1120&upscale=true&name=Artboard%20215_5.png',
            'Stress': 'https://9251305.fs1.hubspotusercontent-na1.net/hub/9251305/hubfs/Artboard%20215_4.png?width=1120&upscale=true&name=Artboard%20215_4.png',
        }
        html = f"""
<div id="hs_cos_wrapper_module-5-0-0" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module" style="color: inherit; font-size: inherit; line-height: inherit;" data-hs-cos-general-type="widget" data-hs-cos-type="module">









<!--[if gte mso 9]>
<v:rect xmlns:v="urn:schemas-microsoft-com:vml" fill="true" stroke="false" style="width:600px; height:2pt;" fillcolor="none">
<v:fill type="tile"/>
<v:textbox inset="0,0,0,0">

<div>

<![endif]-->
<table role="presentation" style="position:relative; top:-1px; min-width:20px; width:100%; max-width:100%; border-spacing:0; mso-table-lspace:0pt; mso-table-rspace:0pt; border-collapse:collapse; font-size:1px" width="100%" border="0" align="center">
  <tbody><tr>
    
    
    
    <td style="border-collapse:collapse; mso-line-height-rule:exactly; font-family:Arial, sans-serif; font-size:15px; color:#23496d; word-break:break-word; line-height:0; border:transparent; border-bottom:1px solid #99ACC2; mso-border-bottom-alt:1px solid #99ACC2; border-bottom-width:1px" width="100%" valign="middle">&nbsp;</td>
    
    
  </tr>
</tbody></table>

<!--[if gte mso 9]></div></v:textbox></v:rect><![endif]--></div>

<div id="hs_cos_wrapper_module_16536868701967" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module" style="color: inherit; font-size: inherit; line-height: inherit;" data-hs-cos-general-type="widget" data-hs-cos-type="module">




	


<table class="hse-image-wrapper" role="presentation" style="border-spacing:0 !important; border-collapse:collapse; mso-table-lspace:0pt; mso-table-rspace:0pt" width="100%" cellspacing="0" cellpadding="0">
    <tbody>
        <tr>
            <td style="border-collapse:collapse; mso-line-height-rule:exactly; font-family:Arial, sans-serif; color:#23496d; word-break:break-word; text-align:center; padding:10px 20px; font-size:0px" valign="top" align="center">
                
                <img alt="Artboard 215-1" src="{images[condition_name]}" style="outline:none; text-decoration:none; -ms-interpolation-mode:bicubic; max-width:100%; font-size:16px" width="560" align="middle">
                
            </td>
        </tr>
    </tbody>
</table></div>


<div id="hs_cos_wrapper_module_165369182835011" class="hs_cos_wrapper hs_cos_wrapper_widget hs_cos_wrapper_type_module" style="color: inherit; font-size: inherit; line-height: inherit;" data-hs-cos-general-type="widget" data-hs-cos-type="module">




	


<table class="hse-image-wrapper" role="presentation" style="border-spacing:0 !important; border-collapse:collapse; mso-table-lspace:0pt; mso-table-rspace:0pt" width="100%" cellspacing="0" cellpadding="0">
    <tbody>
        <tr>
            <td style="border-collapse:collapse; mso-line-height-rule:exactly; font-family:Arial, sans-serif; color:#23496d; word-break:break-word; text-align:center; padding:10px 20px; font-size:0px" valign="top" align="center">
                
                    <img alt="Plot" src="{image_link}" style="outline:none; text-decoration:none; -ms-interpolation-mode:bicubic; max-width:100%; font-size:16px" width="560" align="middle">
                     
                
            </td>
        </tr>
    </tbody>
</table></div>        """
        return html

    sections = [generate_one_section_code(*x) for x in sections]
    sections = '\n'.join(sections)
    return sections


if __name__ == '__main__':
    formatter_ex = {
        'first_name': 'Jane Doe',
        'email': 'leo@radiclescience.com',
        'conditions':
            {
                'Anxiety': {
                    'condition_valid': True,
                    'invert': True,  # means HIGHER scores are BETTER
                    'scores_cohort': [4.4, 5.2, 5.3, 4.9, 5.5],
                    'scores_individual': list(reversed([5, 6, 7, 8, 7]))
                },

                'Wellbeing': {
                    'condition_valid': True,
                    'invert': False,  # means LOWER scores are BETTER
                    'scores_cohort': [12, 11, 13, 12, 13],
                    'scores_individual': [13, 11, 14, 15, 16]
                },

                'Sleep': {
                    'condition_valid': True,
                    'invert': True,  # means HIGHER scores are BETTER
                    'scores_cohort': [5, 7, 10, 8],
                    'scores_individual': [5, None, 3.5, 4]
                },

                'Pain': {
                    'condition_valid': True,
                    'invert': True,  # means HIGHER scores are BETTER
                    'scores_cohort': [5, 7, 6, 8],
                    'scores_individual': [5, 4, 3, 4]
                },
            },

        'primary_outcome': '\<study condition, ex. "Pain"\>',
        'comparison_group': 'control',
        'study_size': '2500',
        'top_states': 'CA and TX',
        'top_demographic': 'Females over 40',
    }
    ret = generate_from_document(formatter_ex, savepath='test_b64.html')
    print('Done Testing')
