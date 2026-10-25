from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

texts = ["Win free cash prize now", "Claim free lottery money", "Win free gift card",
         "Meeting at 2pm tomorrow", "Send the project report", "Lunch with team today"]
y = [1, 1, 1, 0, 0, 0] # 1 = Spam, 0 = Ham

X = CountVectorizer().fit_transform(texts)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

model = MultinomialNB().fit(X_tr, y_tr)
pred = model.predict(X_te)

print("Accuracy:", accuracy_score(y_te, pred))
print("Report:\n", classification_report(y_te, pred, zero_division=0))