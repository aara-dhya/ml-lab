from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

X = ["win money", "hello friend", "claim prize", "good morning"]
y = ["Spam", "Not Spam", "Spam", "Not Spam"]

v = CountVectorizer()
X = v.fit_transform(X)
m = MultinomialNB().fit(X, y)

print(m.predict(v.transform(["win prize"])))