from models import wrappers
from goal_functions import UntargetedClassification
from transformations import WordSwapEmbedding
# Load model, tokenizer, and model_wrapper
model_wrapper = wrappers.LLMModelWrapper(prompts='What is the sentiment of the following review?', labels=['Negative', 'Positive'])

# Construct our four components for `Attack`
from constraints.pre_transformation import RepeatModification, StopwordModification
from constraints.semantics import WordEmbeddingDistance

goal_function = UntargetedClassification(model_wrapper)
constraints = [
    RepeatModification(),
    StopwordModification(),
    WordEmbeddingDistance(min_cos_sim=0.9)
]
transformation = WordSwapEmbedding(max_candidates=50)
search_method = GreedyWordSwapWIR(wir_method="delete")

# Construct the actual attack
attack = Attack(goal_function, constraints, transformation, search_method)

input_text = "I really enjoyed the new movie that came out last month."
label = 1 #Positive
attack_result = attack.attack(input_text, label)

# import transformers
# from models import wrappers
# # Load model, tokenizer, and model_wrapper
# model = transformers.AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
# tokenizer = transformers.AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
# model_wrapper = models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

# # Construct our four components for `Attack`
# from constraints.pre_transformation import RepeatModification, StopwordModification
# from constraints.semantics import WordEmbeddingDistance

# goal_function = goal_functions.UntargetedClassification(model_wrapper)
# constraints = [
#     RepeatModification(),
#     StopwordModification(),
#     WordEmbeddingDistance(min_cos_sim=0.9)
# ]
# transformation = WordSwapEmbedding(max_candidates=50)
# search_method = GreedyWordSwapWIR(wir_method="delete")

# # Construct the actual attack
# attack = Attack(goal_function, constraints, transformation, search_method)

# input_text = "I really enjoyed the new movie that came out last month."
# label = 1 #Positive
# attack_result = attack.attack(input_text, label)