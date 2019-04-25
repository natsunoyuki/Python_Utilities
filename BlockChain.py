import hashlib
import datetime as date
import random 
import string

"""
An example of an application of the blockchain technology in python from:
https://medium.com/crypto-currently/lets-build-the-tiniest-blockchain-e70965a248b
for a "crytocurrency" called SnakeCoin.

This code was obtained from the URL above, with some personal applications
of my own.

Blockchain is a digital ledger in which transactions made in bitcoin or another 
cryptocurrency are recorded chronologically and publicly.
"""

class Block:
    def __init__(self,nonce,timestamp,data,previous_hash):
        self.nonce=nonce
        self.timestamp=timestamp
        self.data=data
        self.previous_hash=previous_hash
        self.hash=self.hash_block()
        
    def hash_block(self):
        #Each block is hashed with the following information:
        #nonce + timestamp + data + previous hash
        x=self.nonce+str(self.timestamp)+str(self.data)+str(self.previous_hash)
        return hashlib.sha256(x).hexdigest()
    
    def __str__(self):
        return "{0}, {1}".format(self.timestamp,self.data)

def create_genesis_block(N=4):
    #Manually construct a block with
    #index zero and arbitrary previous hash
    #the genesis block assigns a certain address with a certain value
    #      Block(index, timestamp, data, previous hash)
    starting_nonce=random.choice(string.ascii_lowercase+string.digits)
    starting_hash=hashlib.sha256(starting_nonce).hexdigest()
    Data={"Natsume":100}
    X=Block(starting_nonce,date.datetime.now(),Data,starting_hash)
    while X.hash[:N]!="0"*N:
        starting_nonce+=random.choice(string.ascii_lowercase+string.digits)
        X=Block(starting_nonce,date.datetime.now(),Data,starting_hash)
        #X.previous_hash = hashlib.sha256(X.nonce[0]).hexdigest()
    return X

def next_block(last_block,transaction_details,N=4):
    this_nonce=random.choice(string.ascii_lowercase+string.digits)
    this_timestamp=date.datetime.now()
    this_data=transaction_details.copy()
    this_hash=last_block.hash
    test_block=Block(this_nonce,this_timestamp,this_data,this_hash)
    while test_block.hash[:N]!="0"*N:
        this_nonce+=random.choice(string.ascii_lowercase+string.digits)
        test_block=Block(this_nonce,this_timestamp,this_data,this_hash)
    return test_block

def create_transaction(old_data,addr_1,addr_2,value):
    a=old_data.get(addr_1)-value
    if a<0:
        print "Impossible to pay {0} from {1}!".format(value,addr_1)
        return old_data
    new_data=old_data.copy()
    new_data[addr_1]=a
    new_data[addr_2]=old_data.get(addr_2,0)+value
    return new_data

def pay_coins(payer,payee,amount):
    transaction=create_transaction(MyBlockChain[len(MyBlockChain)-1].data,payer,payee,amount)
    if transaction==MyBlockChain[len(MyBlockChain)-1].data:
        print "Transaction was not processed from {0} to {1}!".format(payer,payee)
    else:
        MyBlockChain.append(next_block(MyBlockChain[len(MyBlockChain)-1],transaction,2))

#genesis block introduces 100 coins into the world in Natsume's account
MyBlockChain=[create_genesis_block(2)]

print "Natsume pays Miki 10 coins"
payer='Natsume'
payee='Miki'
amount=10
pay_coins(payer,payee,amount)

print "Natsume pays Naomi 30 coins"
payer='Natsume'
payee='Naomi'
amount=30
pay_coins(payer,payee,amount)

print "Naomi pays Miki 20 coins"
payer='Naomi'
payee='Miki'
amount=20
pay_coins(payer,payee,amount)

print "Miki pays Yurie 10 coins"
payer='Miki'
payee='Yurie'
amount=10
pay_coins(payer,payee,amount)

print "Yurie tries to pay Natsume 20 coins despite having only 10"
payer='Yurie'
payee='Natsume'
amount=20
pay_coins(payer,payee,amount)

print "Yurie then tries to pay Natsume 10 coins"
payer='Yurie'
payee='Natsume'
amount=10
pay_coins(payer,payee,amount)

for i in MyBlockChain:
    print i
