import textattack

# Load model, tokenizer, and model_wrapper
model_wrapper = textattack.models.wrappers.LLMModelWrapper(prompts='What is the sentiment of the following review?', labels=['Negative', 'Positive'])

# Construct our four components for `Attack`
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.constraints.semantics import WordEmbeddingDistance

goal_function = textattack.goal_functions.UntargetedClassification(model_wrapper)
constraints = [
    RepeatModification(),
    StopwordModification(),
    WordEmbeddingDistance(min_cos_sim=0.9)
]
transformation = WordSwapEmbedding(max_candidates=50)
search_method = GreedyWordSwapWIR(wir_method="delete")

# Construct the actual attack
attack = Attack(goal_function, constraints, transformation, search_method)

input_text = "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side."
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