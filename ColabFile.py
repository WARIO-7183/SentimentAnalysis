from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from transformers import AutoTokenizer , AutoModelForSequenceClassification
from scipy.special import softmax
Model="cardiffnlp/twitter-roberta-base-sentiment-latest"
token = AutoTokenizer.from_pretrained(Model)
model = AutoModelForSequenceClassification.from_pretrained(Model)
example="You broke my car.Well Done"
encoded=token(example,return_tensors='pt')
output=model(**encoded)
score=output[0][0].detach().numpy()
score=softmax(score)
score_dict={'negative' : score[0],'neutral' : score[1],'positive' : score[2]}
print(score_dict)
def Polarity(example):
  encoded=token(example,return_tensors='pt')
  output=model(**encoded)
  score=output[0][0].detach().numpy()
  score=softmax(score)
  score_dict={'negative' : score[0],'neutral' : score[1],'positive' : score[2]}
  return score[0],score[1],score[2]
import ipywidgets as widgets
from IPython.display import display

def detect_sarcasm(text):
    neg, neu, pos = Polarity(text)  
    return {'positive': pos, 'neutral': neu, 'negative': neg}


def analyze_text(button_click):
    text = text_input.value
    if text:
        results = detect_sarcasm(text)
        
       
        positive_output.value = f"Positive: {results['positive']}"
        neutral_output.value = f"Neutral: {results['neutral']}"
        negative_output.value = f"Negative: {results['negative']}"
        
       
        positive_output.style.background_color = 'lightgreen'
        neutral_output.style.background_color = 'lightyellow'
        negative_output.style.background_color = 'lightcoral'


text_input = widgets.Text(
    value='',
    placeholder='Type something...',
    description='Text:',
    disabled=False
)


analyze_button = widgets.Button(description="Analyze")
analyze_button.on_click(analyze_text)


positive_output = widgets.Label(value="Positive: ")
neutral_output = widgets.Label(value="Neutral: ")
negative_output = widgets.Label(value="Negative: ")


display(text_input, analyze_button, positive_output, neutral_output, negative_output)
