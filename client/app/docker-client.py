import warnings

warnings.filterwarnings('ignore')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import io
import requests as rq
import json
import secrets
import time
import base64

from random import randrange
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from sslib import shamir, randomness
import numpy as np
import pickle
# from modelTf import ModelTf
from model import ModelTf
from timeMetrics import timeMetrics
import pandas as pd 

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Build simple model
CLIENTNUMBER = 3


MODEL = ModelTf()
TimeMetrics = timeMetrics()
# CONST
SERVER_IP = 'http://localhost:3000'  # os.environ['SERVER_IP'] or
PULL_REQUEST_INTERVAL = 2  # in second(s)
HEADERS = {'Content-type': 'application/json'}
THRESHOLD = None
BYTES_NUMBER = 4
# MODE = 3

CLIENT_LIST = []
modelMatrix_Dict = {}
CUSK_Dict = {}
SuSK_Dict = {}
U1_Dict = {}
U2_Dict = {}
bu_Dict = {}
RandomMatric_Dict = {}
EncEntropy_Dict = {}

############## MODEINIT ##############

# Get current try
res = rq.get(SERVER_IP + '/tries/current')
initialParams = res.json()
idTry = str(initialParams['idTry'])

# Get initial params
for i in range(1, CLIENTNUMBER+1):
    res = rq.get(SERVER_IP + '/tries/' + idTry + '/initial-params')
    initialParams = res.json()
    threshold = initialParams['threshold']
    idUser = initialParams['idUser']
    CLIENT_LIST.append(idUser)
    # print(idUser)

# Get model update
modelidx = 0
for idUser in CLIENT_LIST:
    res = rq.get(SERVER_IP+'/model')
    r = base64.b64decode(res.text)
    initialTrainableVars = np.frombuffer(r, dtype=np.dtype('d'))
    MODEL.updateFromNumpyFlatArray(initialTrainableVars)    

    # randomIndex = randrange(0,25)
    print('Client ' + str(idUser) + ' will train with part ' + str(modelidx) + ' of the dataset')

    acc = MODEL.trainModel(5, modelidx)
    print('Client ' + str(idUser) + ' accuracy after training (with test values): ' + str(acc))
    modelMatrix = MODEL.toNumpyFlatArray().copy()
    # print(modelMatrix.shape)
    filename = 'app/updates/plaintext' + str(modelidx) + '.csv'
    df = pd.DataFrame([modelMatrix])
    file_exists = os.path.isfile(filename)
    df.to_csv(filename, mode='a', index=False, header=not file_exists)
    modelMatrix_Dict[idUser] = modelMatrix
    modelidx += 1 


start_time = time.time()
# TimeMetrics.addTime('Start')
# for idUser in CLIENT_LIST:
#     acc, _, _ = MODEL.trainModel()
#     modelMatrix = MODEL.toNumpyFlatArray().copy()
#     modelMatrix_Dict[idUser] = modelMatrix
training_time = time.time()
# TimeMetrics.computeTime('Training-time', training_time-start_time)


#################### STEP 0  Initialization ############################

for idUser in CLIENT_LIST:
    CuSK = ec.generate_private_key(
        ec.SECP384R1(),
        default_backend())
    CuPK = CuSK.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    SuSK = ec.generate_private_key(
        ec.SECP384R1(),
        default_backend())
    SuPK = SuSK.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    keys = {'CuPK': str(CuPK, "utf-8"),
            'SuPK': str(SuPK, "utf-8")}
    CUSK_Dict[idUser] = CuSK
    SuSK_Dict[idUser] = SuSK
    keys_json = json.dumps(keys)
    # Send public keys to server
    res = rq.post(SERVER_IP + '/tries/' + idTry + '/rounds/0/public-keys?userId=' + str(idUser), data=keys_json,
                  headers=HEADERS)



for idUser in CLIENT_LIST:
    # Get U1 the list with public keys from all clients
    url = SERVER_IP + '/tries/' + idTry + '/rounds/1/public-keys'
    res = rq.get(url)
    while res.status_code != 200:
        res = rq.get(url)
        time.sleep(PULL_REQUEST_INTERVAL)
    clientsU1 = res.json()
    U1_Dict[idUser] = clientsU1

print('U1length:', len(clientsU1))



for idUser in CLIENT_LIST:
    # Generate random client mask vector bu
    bu = secrets.token_bytes(16)
    bu_Dict[idUser] = bu
    # Split bu into n (=client numbers) shares
    shamirResBu = shamir.split_secret(
        bu,
        threshold-1,
        len(clientsU1) - 1,
        randomness_source=randomness.UrandomReader()
    )
    # Encode shares with base64
    shamirResBu = shamir.to_base64(shamirResBu)

    sharesBu = shamirResBu['shares']

    # Splitting SuSK into n (=client numbers) shares
    SuSK_as_string = SuSK_Dict[idUser].private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    shamirResSuSK = shamir.to_base64(shamir.split_secret(SuSK_as_string, threshold-1, len(clientsU1) - 1))
    sharesSuSK = shamirResSuSK['shares']
    # print(sharesBu, sharesSuSK)
    ciphertexts = []
    i = 0

    # Compute cipher text for each client
    clientsU1 = U1_Dict[idUser]
    for client in clientsU1:
        if client['id'] != idUser:
            toBeEncrypted = str(idUser) + ';' + str(client['id']) + ';' + sharesSuSK[i] + ';' + sharesBu[i]
            # Parse client Cu public key
            publicKey = serialization.load_pem_public_key(client['publicKeyCu'].encode('utf-8'), default_backend())
            # Create shared key
            sharedKey = CUSK_Dict[idUser].exchange(ec.ECDH(), publicKey)
            # Perform key derivation and encode key
            derivedKey = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=None,
                backend=default_backend()
            ).derive(sharedKey)
            key = base64.urlsafe_b64encode(derivedKey)
            f = Fernet(key)
            ciphertexts.append({
                'id': client['id'],
                'ciphertext': str(f.encrypt(toBeEncrypted.encode('utf8')), "utf-8")
            })
            # Save fernet instance for future use (round 4)
            client['fernet'] = f
            i += 1
    U1_Dict[idUser] = clientsU1

    reqData = {
        'suSKPrimeMod': shamirResSuSK['prime_mod'],
        'buPrimeMod': shamirResBu['prime_mod'],
        'ciphertexts': ciphertexts
    }

    # print(reqData)
    reqJson = json.dumps(reqData)
    res = rq.post(SERVER_IP + '/tries/' + idTry + '/rounds/1/ciphertexts?userId=' + str(idUser), data=reqJson,
                  headers=HEADERS)


# Get U2 the list of ciphertexts from all clients
for idUser in CLIENT_LIST:
    url = SERVER_IP + '/tries/' + idTry + '/rounds/2/ciphertexts?userId=' + str(idUser)
    res = rq.get(url)
    while res.status_code != 200:
        res = rq.get(url)
        time.sleep(PULL_REQUEST_INTERVAL)
    clientsU2 = res.json()
    U2_Dict[idUser] = clientsU2
    print('U2length:', len(clientsU2))







# We are going to use AES in CTR mode as a pseudo random generator
# to generate Puv & Pu
# CTR is configured with full zero nounce
# AES will encrypt full zero plaintext every time but using different key
id = 0
for idUser in CLIENT_LIST:
    maskedVector = modelMatrix_Dict[idUser] * 10 ** 8
    for i in range(len(maskedVector)):
        maskedVector[i] = int(maskedVector[i])
    # maskedVector = np.array([int(item) for item in maskedVector])
    clientsU2 = U2_Dict[idUser]
    ctr = modes.CTR(b'\x00' * 16)
    initialPlaintext = b'\x00' * BYTES_NUMBER * maskedVector.size

    random_matrix = np.zeros(24, dtype=np.dtype('i4'))

    for client in clientsU2:

        if client['id'] == idUser:
            raise Exception('ROUND2: idUser schouldn\'t be equal to received client')
        # Parse client Su public key
        SuPkClient = serialization.load_pem_public_key(client['publicKeySu'].encode('utf-8'), default_backend())

        # Create shared key
        sharedKey = SuSK_Dict[idUser].exchange(ec.ECDH(), SuPkClient)
        # Perform key derivation
        derivedKey = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=None,
            backend=default_backend()
        ).derive(sharedKey)

        # **** COMPUTE Puv (client shared mask)

        cipherPuv = Cipher(algorithms.AES(derivedKey), ctr, backend=default_backend())
        encryptor = cipherPuv.encryptor()

        # Generate random bytes to fill Puv array
        ct = encryptor.update(initialPlaintext) + encryptor.finalize()

        # Convert random bytes to a numpy array
        puv = np.frombuffer(ct, dtype=np.dtype('i4'))

        # Compute delta for current client id
        delta = 1 if client['id'] < idUser else -1

        # Add Puv to the maskedVector
        maskedVector += delta * puv
        # print(puv)

    # **** COMPUTE Pu (client personal mask)
    cipherPu = Cipher(algorithms.AES(bu_Dict[idUser]), ctr, backend=default_backend())
    encryptor = cipherPu.encryptor()

    # Generate random bytes to fill Pu array
    ct = encryptor.update(initialPlaintext) + encryptor.finalize()

    # Convert random bytes to a numpy array
    pu = np.frombuffer(ct, dtype=np.dtype('i4'))
    maskedVector += pu
    RandomMatric_Dict[idUser] = maskedVector

    filename = 'app/updates/masked' + str(id) + '.csv'
    df = pd.DataFrame([maskedVector])
    file_exists = os.path.isfile(filename)
    df.to_csv(filename, mode='a', index=False, header=not file_exists)
    id+=1

    pu_bias_index = pu.shape[0]



############################ STEP 3 Proof of Knowledge of Masked Weighted Model (PoKMWM)  ############################
for idUser in CLIENT_LIST:
    # modelMatrix = modelMatrix_Dict[idUser]
    # random_matrix = RandomMatric_Dict[idUser]

    # maskedVector = []
    # for i in range(0, modelMatrix.shape[0]):
    #     element = modelMatrix[i] + random_matrix[i]
    #     # print(element)
    #     maskedVector.append(element)
    # n_json = json.dumps({
    #     'maskedModel': maskedVector
    # })
    # res = rq.post(SERVER_IP + '/tries/' + idTry + '/rounds/3/masked-model?userId=' + str(idUser), data=n_json,
    #               headers=HEADERS)
    maskedVectorEncoded = base64.b64encode(RandomMatric_Dict[idUser])
    res = rq.post(SERVER_IP+'/tries/'+idTry+'/rounds/2/masked-vector?userId='+str(idUser), data=maskedVectorEncoded)
    vectorEncoded = base64.b64encode(modelMatrix_Dict[idUser])
    # print(modelMatrix_Dict[idUser])
    res = rq.post(SERVER_IP+'/tries/'+idTry+'/rounds/2/original-vector?userId='+str(idUser), data=vectorEncoded)

    # reqData = {
    #     'model': modelMatrix_Dict[idUser],
    # }

    # # print(reqData)
    # reqJson = json.dumps(reqData, cls=NumpyEncoder)
    # res = rq.post(SERVER_IP + '/tries/' + idTry + '/rounds/2/vector?userId=' + str(idUser), data=reqJson,
    #               headers=HEADERS)




############################ STEP 4 Weighted Aggregation (WAgg)  ############################

# Get U4 the list of ciphertexts from all clients
for idUser in CLIENT_LIST:
    url = SERVER_IP + '/tries/' + idTry + '/rounds/4/user-list?userId=' + str(idUser)
    res = rq.get(url)

    while res.status_code != 200:
        res = rq.get(url)
        time.sleep(PULL_REQUEST_INTERVAL)

    clientsU4 = res.json()

print('U4length:', len(clientsU4))


for idUser in CLIENT_LIST:
    resClientU4 = []
    clientsU2 = U2_Dict[idUser]
    clientsU1 = U1_Dict[idUser]
    for clientU2 in clientsU2:
        clientU1 = next((clientU1 for clientU1 in clientsU1 if clientU1["id"] == clientU2['id']), None)

        decryptedCipher = str(clientU1['fernet'].decrypt(clientU2['ciphertext'].encode('utf8')), 'utf8')
        decryptedCipher = decryptedCipher.split(';')

        clientU4 = next((clientU4 for clientU4 in clientsU4 if clientU4["id"] == clientU2['id']), None)

        if clientU4:
            resClientU4.append({
                'id': int(clientU2['id']),
                'buShare': decryptedCipher[3],
                'suSKShare': None,
            })
        else:
            resClientU4.append({
                'id': int(clientU2['id']),
                'buShare': None,
                'suSKShare': decryptedCipher[2],
            })
    resClientU4Json = json.dumps(resClientU4)
    res = rq.post(SERVER_IP + '/tries/' + idTry + '/rounds/4/shares?userId=' + str(idUser), data=resClientU4Json,
                  headers=HEADERS)

# Send either a share of SuSK or

wagg_time = time.time()
# TimeMetrics.computeTime('WAgg', wagg_time - training_time)

end_time = time.time()
runtime = end_time - training_time
# TimeMetrics.computeTime('Runtime', runtime)
# TimeMetrics.addTime('End')
# TimeMetrics.to_csv()
# TimeMetrics.printTime()
sum = np.zeros(28)
for idUser in CLIENT_LIST:
    sum += np.array(modelMatrix_Dict[idUser])
print(sum/3)