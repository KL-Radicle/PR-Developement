import boto3
from botocore.exceptions import ClientError


def send_email_reminder(email, message='Test from Radicle', sender='Radicle Science <help@radiclescience.com>',
                        Subject="Radicle Study Reminders", file=None):
    if file is not None:
        try:
            message = open(file, 'r').read()
        except:
            print(f"Can't read file for email input, {file}")

    # email = 'kaus@radiclescience.com'
    client = boto3.client('ses', region_name="us-west-2")
    SENDER = sender
    RECIPIENT = email

    try:
        # Provide the contents of the email.
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            Message={
                'Body': {
                    'Html': {
                        'Charset': "UTF-8",
                        'Data': message,
                    },
                    'Text': {
                        'Charset': "UTF-8",
                        'Data': message,
                    },
                },
                'Subject': {
                    'Charset': "UTF-8",
                    'Data': ("%s" % Subject),
                },
            },
            Source=SENDER,
            # If you are not using a configuration set, comment or delete the
            # following line
            # ConfigurationSetName=CONFIGURATION_SET,
        )
    # Display an error if something goes wrong.
    except ClientError as e:
        print(f"error: {e.response['Error']['Message']}")
    else:
        print(f"Email sent! Message ID: {response['MessageId']}"),

    return response


send_email_reminder('kaus@radiclescience.com', Subject='Your Radicle Health Journey Report (Sample)',
                    file='./test_b64.html')
