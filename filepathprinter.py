import os

for filename in os.listdir("images/ExtrasolarObjects/"):
    filepath = "images/ExtrasolarObjects" + "/" + filename

    print(f"\'{filepath}\':")