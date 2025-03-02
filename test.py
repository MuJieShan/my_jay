from transformers import pipeline,AutoModelForSequenceClassification
a=1.0e-7
b=1.0e-8
c=a+2*b
d=1.1e-7
r=[a+i*b for i in range(0,10)]
print(a,b,c,d)
print(r)
