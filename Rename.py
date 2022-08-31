import os

os.chdir("D:\\Mirror\\mphil\\Cheryl's folder\\go pro\\Stanley_2")

wrong_named = [f for f in os.listdir() if "xml" in f]

for f in wrong_named:
    os.rename(f, f.replace(".xml", ""))