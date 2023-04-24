from nltk.tokenize import sent_tokenize
import spacy
from spacy import displacy

# Load the language model
nlp = spacy.load("en_core_web_sm")
'''
# Add the sentencizer component to the pipeline --
sentencizer = nlp.add_pipe("sentencizer")
nlp.add_pipe(sentencizer, before="parser")
'''
#read file contents
f = open('cinderella_story.txt', 'r')
document = f.read()
f.close()

# Tokenize the text into sentences
sentences = sent_tokenize(document)

# Parse each sentence using the model's dependency parser
for sentence in sentences:
    doc = nlp(sentence)
    html = displacy.render(doc, style='dep', jupyter=False, options={'distance': 120})
    # Save the HTML to a file or view it in a text editor
    with open("parse_tree.html", "a", encoding="utf-8") as f:
        f.write(html)

#sentence = 'Bibbidi Bobbidy Boo!'

# nlp function returns an object with individual token information,
# linguistic features and relationships

'''
print("{:<15} | {:<8} | {:<15} | {:<20}".format(
    'Token', 'Relation', 'Head', 'Children'))
print("-" * 70)

for token in doc:
    # Print the token, dependency nature, head and all dependents of the token
    print("{:<15} | {:<8} | {:<15} | {:<20}"
          .format(str(token.text), str(token.dep_), str(token.head.text), str([child for child in token.children])))

   # Use displayCy to visualize the dependency
    html = displacy.render(doc, style='dep', jupyter=False, options={'distance': 120})
    # Save the HTML to a file or view it in a text editor
    with open("parse_tree.html", "w", encoding="utf-8") as f:
        f.write(html)
'''