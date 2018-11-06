# General Q&A
# [Deep Learning]
#### Question1. 머신러닝이 할수 있는 예측에 대하여 궁금합니다. 일부에서는 스포츠 승률이라던지 심지어는 복권당첨도 예측하던데요.

    머신 러닝은 짧은 시간 동안에 여러 분야에 극적인 영향을 끼치고 있습니다. 예측 할 수 있는 문제의 유형도 시간이 지나면서 Combinatorial하게 넓어지고 있는 것 같습니다. 

    승포츠 승률 뿐만아니라, 기존에 연구되어 왔던 거의 모든 주제에 있어서 새로운 방법론의 등장으로 인해 새롭게 다루어 지고 있는 것 같습니다. 

    (단, 얘기 주신, 복권 당첨의 경우는 독립된 환경에서 random하게 추출한 결과일텐데,
     단기적으로는 마치 어떠한 요인이 결과와 correlation이 있는 것처럼 보일 수 있어도 spurious correlation일 것 입니다.)

    개인적으로 제 주변에서 재밌게 하고 있는 연구들은, 
        1. (의료 분야) 암 환자들의 모바일 사용 정도로 우울증 상태를 예측
        2. (의료 분야) 쥐의 똥에서 추출한 데이터에 기반해서 쥐의 장 내부에 있는 모든 박테리아의 종류와 양을 예측
        3. (경영학 분야) Airbnb에서 어떤 사진을 찍어 올려야 수입이 높은지에 대한 예측
        4. (경영학 분야) 어떤 식으로 Speech를 한 벤처 기업가가 성공적으로 투자를받을 수 있을지 예측
        5. (경영학 분야) 어떤 식으로 Hand Gesture를 보여야 인터넷 강의의 등록률이 높아질지 예측        
    등이 있는 것 같습니다. 이외에도 나열하자면 끝이 없는 예측 문제들이 있습니다.

    현실에서의 Application이 궁금하시면, Top 10 Applications of Machine Learning (https://www.youtube.com/watch?v=ahRcGObyEZo)을 참고해 보시면 좋을 것 같습니다.

    혹시 경제학 분야에서의 적용이 궁금하시면, Stanford의 Susan Athey 교수의 "The Impact of Machine Learning on Economics" 논문을 읽어보시길 권유드립니다. (https://www.nber.org/chapters/c14009.pdf)
    
#### Question2. 과연 딥러닝으로 금융시장을 이길수있는 전략을 도출할수있을까요?

    효율적 시장가설에 따르면 알려진 정보로는 수익(alpha)을 낼 수 없습니다. 따라서, 다른 사람에 비해서 Superior한 정보, 방법론 등을 가지고 있는 것이 중요한 것 같습니다. 

    주식에서 주가 예측 모형으로 가장 유명한 Fama French 3 Factor 모델을 예로 들자면,
    해당 모델을 통해서 과거 몇년 동안은 예측률이 있고 수익이 났지만, 해당 모델이 널리 알려진 뒤에는 
    더 이상 다른 사람에 비해 Superior한 모델이 아니기 때문에 수익이 관측되지 않는다 라는 논문이 있습니다.

    이와 유사하게, ["Does Academic Research Destroy Stock Return Predictability? (2015)"](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.12365) 논문을 읽어보시길 추천드립니다.

    또한 주식에서는 큰 주식은 이미 너무나 효율적이여서 예측 모델로 수익을 내기가 쉽지 않고, 
    작은 주식은 거래량이 적어서 높은 슬리피지로 인해 수익을 만들어 내기 어려운 점이 있습니다. 
    그래서, 중간 사이즈 크기의 주식에서 오히려 Back test시 수익이 나는 경우도 있는 것 같습니다. 
    코인의 경우도 마찬가지로 BTC에서는 이미 여러 시장 참가자들에 의해 Alpha가 사라지고,
    중간 사이즈의 코인에서의 수익률이 더 나아보기도 합니다. 

    그런데, 예측 모델의 Accuracy Level을 다른 사람의 모델에 비해서 훨씬 더 끌어올릴 수가 있다면, 
    (새로운 Factor 사용, 다른 모델 사용)
    우월한 방법론으로 Alpha를 찾을 수 있지 않을까 기대하고 있습니다.

    하지만, 주의하셔야 할점은 M1.2에서 다루었듯이, Theory 없이 (Causual Relationship) 잘 못된 Factor에 기반한 예측은 심각한 Bias를 만들어 낼 수도 있기 때문에 주의가 필요한 것 같습니다.

#### Question3. Data 형태 (이미지, 동영상, 시계열 등)에 따른 Popular Architecture들을 알 수 있을까요? (Work-in-Progress)
- [For Computer Vision Task](docs/computer_vision.md)
    1. Classification
    2. Object Detection
    3. Instance Segmentation
    4. Other Tasks

- [For Natural Language Processing Task](docs/natural_language_processing.md)
    1. Word Similarities
    2. Text Classification
    3. Neural Machine Translation
    4. Chatbots / Dialog Agents
    5. Question Answering

- [For Voice Related Task](docs/voice.md)
    1. Speech Recognition
    2. Generation of Voice
    3. Lip Reading
    4. Lip Sync from Audio
    
- [Generative Adversarial Networks](docs/generative_adversarial_model.md)
    1. Changing face age with GANs
    2. Professional photos
    3. Synthesization of an image from a text description

- [Reinforcement Learning](docs/reinforcement_learning.md)
    1. Reinforcement training with uncontrolled auxiliary tasks
    2. Learning robots
    3. Learning on human preferences
    4. Movement in complex environments

#### References
- https://arxiv.org/pdf/1708.02709.pdf
- http://web.stanford.edu/class/cs224n/
- https://www.analyticsvidhya.com/blog/2017/08/10-advanced-deep-learning-architectures-data-scientists/
- https://blog.statsbot.co/deep-learning-achievements-4c563e034257
        
# [Keras]
#### Question1. 텐서플로우에서 모델 저장하면 checkpoint 파일등 여러개가 만들어지잖아요. 케라스에서는 .h 파일 하나만 생기는겁니까? 혹시 텐서플로우의 .pb 파일에 대응하는 건가요?

    Checkpointing 기능은 긴 시간 (days, weeks) 동안의 학습시, train과정에 문제가 있더라도 모든 학습 정보를 잃지 않도록하는데 필수적인 기능이여서, 장기적으로는 꼭 배워둘 필요가 있는 것 같습니다. 

    질문 주신 Keras의 경우에는 아래의 callback API를 통해 checkpointing 기능을 제공합니다. 
    해당 기능을 통해서, 저장 위치, checkpoint 이름 (epoch number 혹은 metric으로도 naming 가능), 
    어떠한 조건 (save_best_only, save_weights_only, period)을 만족하는 checkpoint를 생성 할지를 설정할 수 있습니다.
    https://keras.io/callbacks/#modelcheckpoint

    Google Colab에서는 얘기주신 것처럼 training 이후 HDF5 format으로 전체 모델을 직접 저정할 수도 있으며, 
    callback API를 통해 training 과정 중에 모델을 저장하는 것도 가능합니다. 
    자세한 checkpoint 저장 방법은 아래의 튜토리얼을 참조해 주세요. 
    https://colab.research.google.com/github/tensorflow/models/blob/master/samples/core/tutorials/keras/save_and_restore_models.ipynb#scrollTo=2S4xrNJRilwi

#### Question2. keras에서 Build Model부분이 마치 black box 같아 보입니다. 모델 학습에도 변수들의 변화되는 과정을 print 해볼만한 방법이 있을까요? 

    neural net의 디버깅은 원인이 너무나 다양할 수 있어서 (exploding gradients, dead relu, vanishing gradient etc) 항상 쉬운 일이 아닌것 같습니다. 
    저는 주로 model compile 이후에 "model.summary()" 를 통해 전체 shape 구조를 보거나, keras model의 특정 layer를 취득하여 해당 layer의 결과를 확인할 수 있는 function을 만들어 사용합니다. 
    
    간략한 설명을 드리자면, 모델 학습 이후에, 보고자하는 Inner Layer의 Input과 Output을 이용햐여 새로운 함수를 만들어 주시면 됩니다.

<pre><code>
# get layers from the trained model
inp = model.layers[0].input
out = model.layers[0].output

# make a function to see an inner layer result
feature_function = K.function([inp], [out])
</code></pre>

    해당 방법에 대한 구체적인 설명은 아래 링크를 참조바랍니다.

    http://laid.delanover.com/debugging-a-keras-neural-network/

    추가적으로 neural network 모델의 일반적인 디버깅 방식은 아래의 링크를 참조해 주시면 좋을 것 같습니다.

    https://medium.com/machine-learning-world/how-to-debug-neural-networks-manual-dc2a200f10f2

#### Question3. hyper parameter 최적화를 위한 설정

    해당 방법은 Module3에서 관련하여 보다 자세히 다룰 예정입니다. 
    (Learning Curves, Batch Normalization, Dropout and Regularization, Continuous Learning, Hyperparameter Search)

#### Question4. Keras Model Optimizer에서 Option (optimizer, loss, metrics) 항목에서, 각 종류별로 어떤 상황에 어떤 옵션을 사용해야 하나요? 그리고, Loss와 Accuracy 값은 어떻게 해석해야 하나요?

    1. Learning Rate
        cost function의 형태에 따라서 적절한 learning rate에 차이가 납니다. cost function이 Flat할 경우 학습의 속도가 느리기 때문에 learning rate를 상대적으로 크게하여 학습 속도를 빠르게 만들 수 있습니다.

        learning rate가 너무 작거나, 클 경우에 cost function에서는 아래와 같이 너무 느리게 학습하거나, overshooting 하는 문제점이 발생할 수 있습니다.

<img src="img/learning_rate_big_small.jpeg" width="60%" align="center">

<img src="img/learningrates.jpeg" width="60%" align="center">

    2. Optimizer
        1. Gradient Descent
        가장 간단한 Optimizer인 gradient descent는 loss function의 gradient의 반대 방향으로 반복하여 weight를 조정해 주면서 loss function을 최소화 하는 파라미터를 찾습니다.

        하지만, weight를 업데이트 하는 과정이, 모든 train 데이터를 이용해 loss를 측정한 뒤 (1 epoch), 평균을 이용하여 단 한번만 일어나기 때문에 slow converge의 문제점이 있습니다.

        2. Stochastic Gradient Descent
        이러한 문제를 해결하기 위해, Stochastic Gradient Descent에서는, 모든 train 데이터가 아닌 하나의 data point를 이용해 weight를 업데이트 하는 SGD 방식이 소개되었습니다.

        하지만 해당 방식은 noise가 크며, 이로 인해 Local Minima에 빠지는 등의 문제점이 있을 수 있습니다.

        3. Mini Batch Gradient Descent
        이러한 문제를 해결하기 위해, 전체 train 데이터를 Mini Batch로 쪼개어, 각 Mini Batch의 Cost를 이용해 weight를 업데이트 해주는 방법이 도입되었습니다.

        Practical하게는, 다양한 batch 사이즈에 대해서, 어떤 size에서 loss의 감소가 가장 빠른지 실험하여 batch size를 결정하기도 합니다.

        일반적으로는 batch size가 너무 크면, 1번에서 업급한 문제와 마찬가지로 slow converge 문제가 있습니다.

        이 외에도 다양한 SGD의 다양한 변형이 소개되었는데, Momentum, NAG, Adagrad etc. 핵심은 어떻게 자주 weight를 업데이트하면서도 noise를 줄일 수 있을지에 대한 문제를 다루고 있습니다.

<img src="img/sgd.gif" width="60%" align="center">

    3. Kernel Initialization
        각 kernel의 초기 값을 어떻게 설정해줄지에 대해 여러 옵션이 있습니다. (zeros, uniform, normal, he_normal, lecun_uniform)

        하지만, 꼭 한 초기화 방식이 항상 더 나은 결과를 보이는 것은 아니며, Very Sensitive하게 결과가 바뀌기도 합니다. 
        똑같은 설정이여도, 새로 돌리면 결과가 다를 수 있기 떄문에, local minima에 빠질 경우 새로 돌려야 하는 경우도 있습니다.

    4. Keras Accuracy Metrics
    
        1. Mean Squared Error: mean_squared_error, MSE or mse
        2. Mean Absolute Error: mean_absolute_error, MAE, mae
        3. Mean Absolute Percentage Error: mean_absolute_percentage_error, MAPE, mape
        4. Cosine Proximity: cosine_proximity, cosine

    5. Keras Classification Metrics
        
        1. Binary Accuracy: binary_accuracy, acc
        2. Categorical Accuracy: categorical_accuracy, acc
        3. Sparse Categorical Accuracy: sparse_categorical_accuracy
        4. Top k Categorical Accuracy: top_k_categorical_accuracy (requires you specify a k parameter)
        5. Sparse Top k Categorical Accuracy: sparse_top_k_categorical_accuracy (requires you specify a k parameter)

    References
    - http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html
    - https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/
    - http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html

#### Question5. keras와 다르게 텐서플로우에서 왜 변수초기화를 안하면 오류가 나는가요?

    tensorflow api는 low level api로 세션에서 사용되는 모든 변수를 초기화해 주어야 합니다. 
    반면에 keras, slim과 같은 high level api는 변수 초기화가 자동으로 되는 것 입니다. 
    Explicit initialization의 장점은 모델을 체크 포인트에서 다시로드 하는 경우에는 오히려, 계산적인 비용이 드는 initialization을 다시 실행하지 않는 이점이 있습니다. 

#### Question6. model.evaluate(X_test, y_test, batch_size=600,verbose=False)의 결과는 Loss인 건가요? accuracy값을 뽑고 싶으면 어떻게 해야하나요..?

    model.compile 시에 metric을 지정하여 주시면 지정된 metric을 계산하며, 그렇지 않을 경우 Loss가 계산됩니다.

#### Question7. 모델저장시 구조 즉 그래프 메타정보 variable 정리가 필요합니다.
    각 layer를 추가하실 때 name을 통해 저장해 주시면, get_layer(name) 을 통해 접근도 가능하며, 메타정보를 관리할 수 있습니다.

# [Data]
#### Question1. Test, Train Set의 할당에 황금(?) 비율이 있는지요? 그리고 예를 들어 data가 충분하지 않을때 Train에 많이 할당하면 좋을텐데, 이런 경우를 위해서 할당량을 조절할 방법이 있나요?

    train / test split의 목적은 generalization error 를 최소화 하기 위함인데, 둘 사이의 샘플 개수 결정은 경쟁 관계에 있습니다. 
    만약 train 사이즈가 작으면 paramter estimate의 variance 가 높고, 반대로 test 사이즈가 작으면 performace statistic의 variance가 높아집니다. 
    따라서, 양쪽의 variance가 너무 높지 않도록 설정해야하는데, 이는 sample size에 따라서 그 크기를 결정하는 방식에 차이가 날 것 같습니다.

    예를들어 100개의 sample에 대해서 train / test 를 쪼갤때는 어떤 방식을 취해도 parameter estimate variance를 줄이기가 쉽지가 않을 것이고, 100000개의 sample을 train / test 를 쪼갤때는 0.8:0.2 를 취하거나, 
    0.9:0.1 을 취하든 generalization에 문제가 없을 것으로 보입니다. 
    어떤 황금률을 제시하기는 어렵지만, variance를 줄이기 위한 방법은 아래와 같은 방법이 있습니다.

        1. training 데이터를 또 다시 train / validation으로 쪼개는 방법
        2. train을 비율 (90, 80, 70) 등을 바꿔가면서 각 비율당 여러차래 (ex. 20회 이상) random sampling 하여 mean performance를 비교하여 lower variance를 선택하는 방법

    저는 약식으로 할때는 보통 70/30 혹은 80/20을 선호합니다. 

    이론적인 부분을 조금 더 살펴보고 싶으시면 아래 논문을 추천드립니다.
    https://pdfs.semanticscholar.org/452e/6c05d46e061290fefff8b46d0ff161998677.pdf

#### Question2. data을 scaling하고 train 하는 이유가 있을것 같아서 문의드립니다.  어떤 모델에서든 항상 0~1값으로 변경해야하는것인지?

    이론적인 면에서는 Normalize는 Neural Network의 Weight, Bias가 조정되는 것으로 모두 잡힐 수 있어야 하지만, Practical 하게는 
        1. training more stable and faster 
        2. reduce the chances of getting stuck in local optima 
        3. 학습과 예측에 사용되는 Domain 데이터 사이의 샘플링 Population 일치 문제 (out of scale from one another)
    를 해결 하기 위한 것으로 알려져 있습니다.

    1번, 2번의 경우를 보다 자세히 얘기드리자면, gradient descent를 할때는 gradient error vectors와 learning rate를 곱해서 weight를 조정해 주는데, 
    input feature들 간의 distribution 차이가 나면, 한쪽 feature에 대해서는 잘 작동하는 learning rate가 다른 feature에 대해서는 under compensating하는 문제가 있을 수 있습니다. 
    (since the learning rate in the update equation of Stochastic Gradient Descent is the same for every parameter) (Elkan, Charles. (2008) Log-linear models and conditional random fields)

    이런 이유로 error surface에서 Global minimum을 찾기 위해서는 구모양에 가까운 input feature를 가지는 것이 타원형 구조 (한쪽 feature의 range가 훨씬 큰 경우)보다 빠르게 converge 하는 것으로 알려져 있습니다.

    3번 같은 경우에 조금 더 쉬운 예를 들자면, 주식/코인 같은 경우에는 Price를 그대로 입력하는 것이 아니라, 
    Return, Log Return을 Scaling의 목적으로 사용할 때가 있는데, 학습할때의 Price 값 (ex: KOSPI: 2000, BTC: $3000)의 기준과 예측할 때(ex: KOSPI: 2300, BTC: $7000)의 Price 값의 기준이 크게 차이가 나는 문제를 해결하기 위해서 사용합니다. 
    (같은 input feature이지만, 시기가 달라서 학습에 관찰되지 않은 데이터가 입력 될 수 있음, out of scale from one another) 

    또한, 가격에 대한 입력과, Bid/Ask Volume에 대한 입력은 Scale 차이가 상당한데 이러할 경우에, 모든 feature에 기반하여 모델이 작동하기 보다는,
    Range가 보다 큰 particular한 feature에 모델이 종속되어 버리는 것 같습니다.  
    (ex. 모델 내에서 distance metric이 계산될 경우 값의 범위가 큰 feature에 종속)

    그리고, 마지막으로 0~1 값으로 변경해야 할지, 다른 옵션 (Mean Normalization, Standardization, Scaling to unit length)를 할지, 
    혹은 Scailing이 하지 않을지 등에 대한 Feature Scaling 선택 문제가 있을 수 있는데, 
    풀고자 하는 문제의 특성에 따라서 다른 Scaling 방법을 취해주는 것이 좋을 것 같습니다. 
    가량 Exact Boundary가 있는 문제의 경우에는 Standardization이 적절하지 않을 수도 있고, 
    min-max scaling은 long tail 분포가 있는 문제에 대해서는 적절하지 않을 수 있습니다. 

#### Question3. 딥러닝에서 레이어를 몇개로 하며 또 각 레이어마다 차원을 몇개로 해야하는지는 어떻게 정하는 겁니까?

    안녕하세요. 좋은 질문 주셔서 감사합니다. 딥러닝에서 레이어의 개수 혹은 차원의 개수에 대한 결정은 Overfitting Issues와 큰 관련이 있습니다. 
    Overfitting은 1. 모델이 너무 복잡하거나 2. 데이터가 너무 적거나 하는 문제로 인해서 발생할 수 있습니다.

    1번의 모델이 너무 복잡해서 발생하는 문제는 주로 Cross Validation을 통해 Overfitting이 발생하지 않았는지를 테스트하며, 같은 결과면 간단한 모델이 더 좋은 모델 (오캄의 면도날)의 원칙을 적용하는 것을 권장합니다. 
    관련한 보다 자세한 이슈는 아래 링크를 통해 공유드립니다. (https://drive.google.com/open?id=1HqCyp94ZC9f7jwhStFnSbuF9TDyrf5iy) 
    Overfitting을 해결하기 위한 방법은 Module3에서 관련하여 보다 자세히 다룰 예정입니다. 
    (Learning Curves, Batch Normalization, Dropout and Regularization, Continuous Learning, Hyperparameter Search)

    2번의 데이터 샘플에 대한 이슈는, 주어진 데이터에 대해 Train 데이터, Test 데이터의 크기를 결정해야 할 때가 있는데, 둘 사이의 샘플 개수 결정은 경쟁 관계에 있습니다. 
    최종적인 목적은, generalization error 를 최소화 하기 위함인데, 만약 train 사이즈가 작으면 paramter estimate의 variance 가 높고, 
    반대로 test 사이즈가 작으면 performace statistic의 variance가 높아집니다. 
    따라서, 양쪽의 variance가 너무 높지 않도록 설정해야하는데, 이는 sample size에 따라서 그 크기를 결정하는 방식에 차이가 날 것 같습니다.

    예를들어 100개의 sample에 대해서 train / test 를 쪼갤때는 어떤 방식을 취해도 parameter estimate variance를 줄이기가 쉽지가 않을 것이고, 
    100000개의 sample을 train / test 를 쪼갤때는 0.8:0.2 를 취하거나, 0.9:0.1 을 취하든 generalization에 문제가 없을 것으로 보입니다. 
    variance를 줄이기 위한 방법으로는 train을 비율 (90, 80, 70) 등을 바꿔가면서 각 비율당 여러차래 (ex. 20회 이상) 
    random sampling 하여 mean performance를 비교하여 lower variance를 선택하는 방법을 쓰기도 합니다.

    Cross Validation에 대한 설명은 아래 링크를 참조해 보시면 좋을것 같습니다. (https://www.youtube.com/watch?v=TIgfjmp-4BA)
    모델이 복잡할때 생길 수 있는 이슈에 대한 예제는 아래의 링크를 참조해 보시면 됩니다. (https://towardsdatascience.com/beginners-ask-how-many-hidden-layers-neurons-to-use-in-artificial-neural-networks-51466afa0d3e)

#### Question4. AI로 뭔가를 학습 시킬 때 전처리를 할텐데 이런 타임시리즈의 데이타는 어떤식으로 전처리를 하나요? 이동평균선?, 볼린져밴드 등 사용하나요?

    좋은 질문 입니다. 얘기 주신 방법 (MA, 볼린저벤드)은 주로 Trend와 Cycle을 구분하기 위한 방법으로 보입니다. 
    사실 이동평균법을 활용하여 Practical하게는 Noise가 있는 Time Series에서 Noise를 줄이기 위해 사용하는 경우도 자주 보지만, 
    이에대한 학문적인 근거를 가지고 사용하는 경우는 주로 보지는 못 한것 같습니다.

    해당 방법과 유사한 방법으로, 주로 학계에서는 Trend와 Cycle을 구분하기 위한 방법으로 Hodrick Prescott Filter를 이용합니다.
    (원레는 Macro Economy에서 Expansion/Recession을 구분하기 위해 사용하던 방법론)

    https://en.wikipedia.org/wiki/Hodrick%E2%80%93Prescott_filter 
    (최근에 흥미롭게 읽었던 논문 중에서는 4차 방정식이 Hodrick Prescott Filter보다 좋으니까 쓰지 말라는 말도 있습니다. https://econweb.ucsd.edu/~jhamilto/hp.pdf)
    
    현재에는 Outcome 변수에 대해서만 설명 드리자면, Preprocessing으로 아래와 같은 과정들을 취하고 있습니다.
    제가 하고있는 Preprocessing에서의 핵심은 데이터의 분포를 가능한 Normal Distribution으로 만들어 주려는 것 이고요, 
    그런 경우에 Significant Change를 잡아내는 방향으로 하고 있습니다. 
    (이외에도 무수히 많은 방법이 가능할 것 같습니다.)

        1. Continuos Value에 대한 예측시에는 Log Return을 예측 값으로 사용 (그냥 Return의 분포는 Kurtosis가 너무 높음)
        3. Continuos Value에 대한 예측시 HP Filter 이후에 Cycle 파트에 대한 예측, Trend 파트에 대한 예측
        2. Discrete Value (Up/Down) 문제를 풀때는, Outcome 변수를 Significant Change를 상정하여 (1 std 이상 상승/횡보/하락)으로 구분

    위의 설명은 Outcome 변수에 대한 것 이지만, 예측에 사용되는 다른 변수들은 그 변수의 특징에 따라서 다른 Preprocessing 과정을 거치고 있습니다.

# [Colab]
#### Question1. Keras와 Google Colab을 사용하는데 가장 적절한 파이썬 버전은 무엇인가요?

    이 부분은 선호의 문제일 수 있을 것 같습니다. 저는 Python2를 주로 쓰고있지만 (legacy code로 인해), 가능하면 Python3를 쓰고 싶습니다.

#### Question2. Colab에서 with tf.device('/gpu:0') 아래에 인덴트해서 넣으면 CPU가 아닌 GPU로 돌아가는 게 맞는지요? 

    안녕하세요 :slightly_smiling_face: 좋은 질문 주셔서 감사합니다. 상단 메뉴의 런타임에서 런타임 유형을 꼭 GPU로 바꿔주셔야 GPU가 작동됩니다. 
    아래 링크에서 참조해 보시면, GPU 설정 이전에는 with tf.device('/gpu:0)을 설정하더라도, 
    GPU 장치가 관측되지 않습니다. (https://colab.research.google.com/notebooks/gpu.ipynb#scrollTo=t9ALbbpmY9rm)

# [Python]
#### Question1. !pip install -q matplotlib-venn 이 뭐하는건지는 알겠는데, -q는 뭘까요?

    아래 링크에 따르면, -q는 설치 과정에 나타나는 메세지를 줄여주는 역할을 합니다.

    -q, --quiet

    Give less output. Option is additive, and can be used up to 3 times (corresponding to WARNING, ERROR, and
    CRITICAL logging levels).

    (https://media.readthedocs.org/pdf/pip/latest/pip.pdf)

#### Question2. 텐서플로우는 64비트에서 돌아가나봅니다. 저는 증권사 API때문에 32비트를 깔은거 같습니다. 이럴 경우에 Keras는 어떻게 설치할 수 있을까요?

    저희 코드에서는 주로 Keras의 Backend로 Tensorflow를 쓰고 있었지만, 32bit에서 설치가 가능한 Theano를 Backend로 가져가셔도 됩니다. 
    혹은 로컬 환경에서의 설치가 까다로우시면, 실습의 진행은 Google Colab을 이용하실 수 있습니다.