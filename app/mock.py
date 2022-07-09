import base64


def mock_audio():

    result =  base64.b64encode("./../record_05_34_54.wav")
    print(result.encode('ascii'))
    
    #return result.decode("utf-8")