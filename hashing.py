#This script shows how hashing is used in proof-of-work in various scenarios.

import hashlib
import string
import random
import time
import uuid

"""
WARNING: in python 2, hashlib takes strings as inputs directly but in python3
the strings must first be encoded to bytes!!!
e.g. b'Hello World!' or str.encode('Hello World!')
"""

def sha256(x):
    if(type(x)!=str):
        x=str(x)
    x=hashlib.sha256(x).hexdigest()
    return x

def proofOfWork(STRING_LENGTH=6):
    N=STRING_LENGTH
    #This function shows a very basic example of proof-of-work explained in
    #https://en.bitcoin.it/wiki/Proof_of_work
    #this creates a random string:
    A=''.join(random.choice(string.ascii_uppercase+string.digits) for _ in range(N))
    B=2**240 #limit value to check the value of the hash against
    C=hashlib.sha256(A).hexdigest() #creates a hexadecimal string
    C=int(C,16) #convert to integer in base 16
    n=0 #iteration counter
    while C>B:
        #add a nonce to the original string and recompute the hash, and check
        #against the limit value
        C=hashlib.sha256(A+str(n)).hexdigest()
        C=int(C,16)
        n=n+1
    print("Needed iterations for '{0}' = {1}".format(A,n))
    
def proofOfWork2(STRING_LENGTH=6,ZEROS_LENGTH=4,TRY_LIMIT=100000000):
    #this function shows another example of proof-of-work where the task is to
    #find another string with at least a certain number of leading zeros.
    N=STRING_LENGTH
    M=ZEROS_LENGTH
    #first create the challenge string A
    A=''.join(random.choice(string.ascii_lowercase+string.digits) for _ in range(N))
    A_orig=A
    D="0"*M #we want to find another string with D leading zeros
    n=0
    STARTTIME=time.time()
    while True:
        #Add a random nonce to the original string and hash it twice. Check
        #if it has the required number of leading zeros
        A=A+random.choice(string.ascii_lowercase+string.digits)
        B=hashlib.sha256(A).hexdigest() 
        B=hashlib.sha256(B).hexdigest()
        n=n+1
        if n%1000000==0:
            print("Currently at {0} tries... time elapsed: ".format(n,time.time()-STARTTIME))
        if n>TRY_LIMIT:
            print("Breaking out of the loop!")
            print("Solution was not found!")
            print("Time elapsed: %.1f s" % (time.time()-STARTTIME))
            break
        if B[0:M]==D:
            print("Needed iterations for '{0}': {1}".format(A_orig,n))
            #print("The nonce found was: {0}".format(A[len(A_orig):]))
            print("The hash found was: {0}".format(B))
            print("Time elapsed: %.1f s" % (time.time()-STARTTIME))
            break
    
def passwordCrack():
    #simple password crack demonstration using a stupid version of brute force
    password='ABC'
    hashed_password=hashlib.sha1(password).hexdigest()
    alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    n=0
    while True:
        guess=alphabet[random.randint(0,len(alphabet)-1)]
        guess=guess+alphabet[random.randint(0,len(alphabet)-1)]
        guess=guess+alphabet[random.randint(0,len(alphabet)-1)]
        hashed_guess=hashlib.sha1(guess).hexdigest()
        n=n+1
        if hashed_guess==hashed_password:
            print("Password cracked!")
            print("{0} tries were taken".format(n))
            break

def hashPassword():   
    #this code was obtained from:
    #https://www.pythoncentral.io/hashing-strings-with-python/
    def hash_password(password):
        # uuid is used to generate a random number
        salt = uuid.uuid4().hex
        return hashlib.sha256(salt.encode() + password.encode()).hexdigest() + ':' + salt
    
    def check_password(hashed_password, user_password):
        password, salt = hashed_password.split(':')
        return password == hashlib.sha256(salt.encode() + user_password.encode()).hexdigest()

    new_pass = raw_input('Please enter a password: ')
    hashed_password = hash_password(new_pass)
    print('The string to store in the db is: ' + hashed_password)
    old_pass = raw_input('Now please enter the password again to check: ')
    if check_password(hashed_password, old_pass):
        print('You entered the right password')
    else:
        print('I am sorry but the password does not match')
        
