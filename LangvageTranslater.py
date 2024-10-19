#from googletrans import Translator
from translate import Translator

#Translator.translate(text="मैं तुम्हें कैसे उठा सकता हूँ",dest="en")

text = Translator(from_lang='Hindi',to_lang='english')

output = text.translate('मैं तुम्हें चोदना चाहता हूँCLS')

print(output)
