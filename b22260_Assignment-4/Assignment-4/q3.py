#3
a = 3
b = 5
i = int(input())

out = ""
for j in range(1,i+1):
    if(out == "buzz"):
        print(j)
    r = j%(a*b)
    if(r == 0):
        out = "fizzbuzz"
    if(r%b == 0):
        out = "buzz"
    if(r%a == 0):
        out = "fizz"
    else:
        out = "null"
    