# Advancing-Cervical-Cancer-Diagnosis-Deep-Learning-Based-Pap-Smear-Analysis
This project aims to enhance cervical cancer diagnosis through deep learning-based analysis of Pap smear images. Cervical cancer, the fourth most common cancer among women, is treatable if detected early. This study utilizes the SIPaKMeD Database, which includes 966 Pap smear cell images categorized as precancerous or normal, to develop a tool that aids doctors in early detection through routine screenings. The project employs a ResNet50-Transformer model, which combines the powerful feature extraction capabilities of a pre-trained ResNet50 convolutional neural network with the attention mechanism of Transformer layers. Data preprocessing involves normalization, standardization, and augmentation techniques to increase training diversity.The model achieves a training accuracy of 96.8%, validation accuracy of 94.3%. This demonstrates its potential to transform cervical cancer screening by reducing reliance on manual processes and improving early detection.

ABSTRACT
Cervical cancer is a widespread health issue that impacts women around the
world. It typically arises after a prolonged period during which precancerous alterations
occur within the cervical tissue. Early detection and timely intervention are
paramount in preventing the progression of precancerous lesions into full-blown
cervical cancer. This study focuses on the development of an advanced diagnostic
tool to aid in the early detection and intervention of cervical cancer. Through
microscopic examination of cells obtained from Pap smear tests, the project explores
the utilization of various deep learning techniques like CNN and Vision
Transformer (ViT) to enhance cervical cancer detection by accurately classifying
cervical cells as either benign or malignant.This paper utilise a novel approach,
a combined ResNet50 and Transformer model, which demonstrates superior performance
in classification. With this approach highest classification accuracy of
92.85% was obtained with ResNet50-Transformer architecture. The implementation
of automated detection methods using deep learning models aims to streamline
diagnostics, thereby reducing reliance on manual screening by pathologists.
Thus by providing healthcare professionals with a reliable and efficient means of
identifying cervical cancer, this research aims to enhance patient outcomes and
prognosis in cervical cancer cases.


1 INTRODUCTION
Cervical cancer ranks as the fourth most common cancer in women worldwide, presenting a significant
global health challenge with profound implications for women’s well-being WHO (2024).
Detecting cervical cancer early is crucial, as it progresses slowly from precancerous changes in cervical
cells to invasive cancer if left untreated. Given its high preventability and curability through
early detection, effective screening and intervention strategies play a pivotal role in lessening the
impact of cervical cancer on women’s health.
Conventional methods of cervical cancer detection, such as Pap smear tests and HPV tests, rely
on microscopic examination of cervical cells for abnormalities. While these screening tests have
proven effective, they have limitations, including the need for specialized training to interpret results
accurately and potential errors in interpretation. Moreover, limited access to screening and follow-up
care in certain regions exacerbates disparities in detection and treatment outcomes.
Despite the availability of screening tests, detecting cervical cancer can be challenging for several
reasons. Precancerous changes in cervical cells often lack noticeable symptoms, necessitating routine
screening for early detection. Interpreting Pap smear results requires specialized training, and
errors can lead to missed diagnoses or false positives. Additionally, access to screening and followup
care may be limited in low-resource settings, contributing to disparities in detection and treatment
outcomes.
Furthermore, factors like variations in the appearance of cervical cells and the presence of other infections
or inflammation can complicate cervical cancer detection. Recent advancements in medical 
imaging and artificial intelligence offer promising opportunities to improve cervical cancer detection.
Deep learning techniques can automate the analysis of Pap smear images to identify abnormal
cells indicative of precancerous or cancerous lesions. By harnessing deep learning capabilities,
healthcare professionals can enhance the accuracy and efficiency of cervical cancer screening, leading
to earlier detection and improved patient outcomes.
This paper explores the emerging field of cervical cancer classification from Pap smear images using
deep learning techniques. The goal is to investigate the potential of the combination of ResNet50 and
ViT models to accurately classify cervical cells as benign or malignant, facilitating early detection of
cervical cancer at the cellular level. By providing an overview of recent advancements, challenges,
and future directions, this study aims to contribute to ongoing efforts to enhance cervical cancer
screening and diagnosis, ultimately reducing the global burden of this disease on women’s health.


2 BACKGROUND
In traditional medical practice, the interpretation of Pap smear images, used for cervical cancer
screening, has been heavily reliant on manual examination by healthcare professionals. However,
this approach has inherent limitations. Firstly, the interpretation can be subjective, as different professionals
may interpret the same image differently, leading to inconsistencies in diagnosis. Additionally,
the manual examination is time-consuming, requiring significant expertise and effort from
healthcare providers.
Manual examination of Pap smear images is time-consuming due to several factors. Healthcare
providers need to carefully scrutinize each image to identify any abnormalities or irregularities in
the cervical cells. Interpreting Pap smear images requires expertise and specialized training. Healthcare
providers need to have a thorough understanding of cervical anatomy, as well as the ability to
differentiate between normal and abnormal cell morphology. Additionally, manual examination of
Pap smear images can be subjective. Different healthcare providers may interpret the same image
differently, leading to inconsistencies in diagnosis. This subjectivity can further contribute to the
time required for examination, as providers may need to consult with colleagues or seek second
opinions to ensure accurate interpretation.
Deep learning models have demonstrated remarkable capabilities in image recognition tasks, making
them well-suited for analyzing complex medical images like Pap smears. These models can
learn intricate patterns and features within images, enabling them to distinguish between normal
cells and potentially cancerous cells with a high degree of accuracy. Integrating such advanced technology
into cervical cancer classification represents a cutting-edge approach that has the potential
to significantly improve diagnostic outcome.
Furthermore, the integration of deep learning into medical image analysis aligns with the broader
trend towards leveraging artificial intelligence in healthcare for enhanced decision-making support.
By automating the analysis of Pap smear images, healthcare providers can streamline the diagnostic
process, reduce human error, and ultimately improve patient outcomes through early detection and
intervention. In conclusion, this project aims to develop a robust and reliable system that can assist
healthcare professionals in accurately identifying cervical abnormalities associated with cancer,
contributing to improved patient care and outcomes in the field of oncology.


3 RELATED WORK
Artificial intelligence and deep learning are crucial in categorizing cells, analyzing medical images,
and aiding in clinical diagnosis within gynecology Wang et al. (2020). As these technologies
advance, they become more cost-effective and time-efficient compared to traditional methods like
Pap smears and cervicography. They are becoming more and more favored over more traditional
methods because they provide an analysis that is devoid of human subjectivity. By providing objective
insights, AI helps healthcare professionals make better clinical decisions, ultimately benefiting
patient care. Attallah (2023) introduced ”CerCan·Net,” a sophisticated computer-aided diagnostic
system specifically tailored for automatic cervical cancer diagnosis. By leveraging lightweight convolutional
neural networks such as MobileNet, DarkNet-19, and ResNet-18, coupled with transfer
learning techniques, CerCan·Net adeptly extracts deep features from diverse layers. Through metic-
ulous feature selection processes, it achieves heightened accuracy in categorizing cervical cancer
subtypes, thus promising substantial support to cytopathologists in enhancing diagnostic precision
and workflow efficiency.
Tripathi et al. (2021) demonstrated the effectiveness of deep transfer learning techniques, particularly
employing ResNet-152 architecture, in analyzing Pap smear images for cervical cancer diagnosis.
While achieving an impressive classification accuracy of 94.89%, the study emphasized the
need to incorporate multiple datasets to enhance model robustness and generalization, as well as
to optimize training runtime to expedite diagnosis processes.Pacal and Kılıcarslan (2023) demonstrated
the utilization of both CNN and vision transformer approaches underscores the versatility of
deep learning methodologies in cervical cancer classification. While CNNs excel in capturing spatial
hierarchies of features, ViT approaches leverage self-attention mechanisms to model long-range
dependencies in images, thus proving effective for tasks where global context is crucial.
Zhao et al. (2022) introduced a novel technique, CCG-taming transformers, which addresses issues
with cervical cell classification, including imbalanced datasets, uneven image quality, and overfitting
in CNN-based models. By combining image generation using transformers with classification using
Tokens-to-Token Vision Transformers, the study achieved a considerable increase in classification
accuracy across various cervical cancer datasets.Park et al. (2021) compared machine learning and
deep learning techniques for cervicography image classification, highlighting the superior performance
of the DL model ResNet-50 over traditional ML methods. However, challenges such as
accurate feature selection and consideration of diagnostic factors identified by multiple model architectures
were identified, suggesting areas for further optimization in computerassisted diagnostic
tools.
Yaman and Tuncer (2022) exemplar pyramid deep feature extraction-based method showcased the
effectiveness of transfer learning-based feature extraction and SVM classification in automatic cervical
cancer detection using Pap smear images.Lastly, Mandelblatt et al. (2002) emphasizes the importance
of well-organized screening programs, particularly in less-developed countries, to significantly
reduce cervical cancer mortality at relatively low costs, while addressing challenges such as ensuring
high follow-up rates and maintaining sensitivity levels in cytology screening.Collectively, these
studies underscore the transformative potential of AI and deep learning in cervical cancer diagnosis
and management, offering insights into innovative techniques, challenges, and opportunities in the
field


4 METHODOLOGY
The primary objective of this research is to explore the potential of advanced models such as
ResNet50 and transformer architectures to develop a highly efficient model for accurately predicting
the cancerous nature of cervical cells. Traditional cervical cancer screening methods, while effective,
are hampered by subjective interpretation and the need for specialized training. By employing
deep learning techniques, the goal is to enhance clinical diagnosis and enable early detection of
cervical cancer, a disease often lacking noticeable symptoms during its precancerous stages. Utilizing
ResNet50 and transformer architectures, the model seeks to utilize rich feature representations
extracted from cervical cell images to effectively differentiate between benign and malignant cells.
This approach not only improves cancer detection accuracy but also simplifies the diagnostic process,
empowering cytologists to make timely and well-informed decisions. Moreover, this research
aims to contribute to the broader goal of reducing the global burden of cervical cancer on women’s
health.
The architecture commences with the pre-trained ResNet50 model as the backbone, renowned for
its exceptional feature extraction capabilities. This backbone architecture comprises 50 layers, pretrained
on the ImageNet dataset, enabling it to extract high-level features effectively from input
images. By using the pretrained ResNet50, the model inherits knowledge learned from a vast array
of images, enhancing its ability to generalize to various visual patterns and structures. Modification
is made to the ResNet50 architecture, specifically the removal of its classification head, traditionally
composed of a fully connected layer followed by a softmax layer for class prediction. This
alteration involves replacing the classification head with an nn.Identity() layer, ensuring that the
ResNet50 solely functions as a feature extractor, generating rich feature representations devoid of
any classification-specific bias.
Following the ResNet50 backbone, a transformative addition is made in the form of a transformer
architecture, known for its proficiency in sequence modeling tasks, notably prevalent in natural
language processing domains. In this context, transformers are adeptly repurposed to process the
extracted image features, expanding the model’s capabilities beyond conventional CNN-based architectures.
The transformation journey begins with an embedding layer, denoted by self.embedding,
which processes the high-dimensional feature vectors outputted by the ResNet50 backbone, compressing
their dimensionality and rendering them conducive for input into subsequent transformer
layers.
The heart of the transformer architecture lies in the transformer encoder layer. This layer incorporates
multiple self-attention mechanisms and feedforward neural networks, facilitating the capture
of intricate relationships and dependencies within the input feature sequence. Parameters such as
’dmodel’ dictate the dimensionality of input and output feature vectors, while nhead governs the
number of parallel attention heads employed in the attention mechanism. Stacked atop the transformer
encoder layer is the transformer encoder, which comprises multiple transformer encoder layers,
typically four in this instantiation. Each layer sequentially processes the input feature sequence,
enabling hierarchical feature extraction and fostering a deeper understanding of the contextual relationships
within the image features.
Finally, a classification head, implemented as a fully connected layer, interprets the transformed
features and maps them to the number of classes pertinent to the classification task. Here, the output
dimensionality is reduced to num classes, facilitating the model’s ability to generate class predictions
based on the learned features. In essence, the ResNet50 Transformer model amalgamates the robust
feature extraction capabilities of ResNet50 with the sequence modeling prowess of transformers,
creating a potent architecture primed for diverse image classification tasks. This synergistic fusion
empowers the model to discern intricate spatial and sequential patterns within images, transcending
the limitations of traditional CNN architectures and yielding heightened classification accuracy and
interpretability.This approach allows leveraging both the powerful feature extraction capabilities of
convolutional neural networks (ResNet-50) and the attention mechanism of Transformer models for
effective classification.


5 EXPERIMENTS
The below block diagram outlines the procedural pipeline for implementing the process.
4

Figure 2: Methodology
5.1 DATASET
This study will be using the SIP Database which is openly accessible and encompasses close to
10,000 isolated cell images extracted from pap-smear samples . These images were captured using
a CCD camera integrated with an optical microscope. Compared to many other datasets utilised in
previously published studies, this dataset is more relevant given all the genetic and environmental
changes that have occurred in the past ten years. The classification was done between five classes
namely, Dyskeratotic, Koilocytotic, Metaplastic, Parabasal and Superficial-intermediate, which can
be categorized into Normal and Abnormal cells.
Figure 3: Different labels in the dataset
Among Normal cells, Superficial-Intermediate cells are characterized by their flattened appearance
with round or oval shapes and typically cyanophilic or eosinophilic cytoplasm, often containing a
centrally thickened or condensed nucleus. Parabasal cells, on the other hand, are small and underdeveloped
epithelial cells found in vaginal smears, displaying cyanophilic cytoplasm and large
vesicular nuclei. The distinction between Parabasal cells and metaplastic cells can be challenging
due to similar morphological features. Abnormal cells, indicating pathological conditions, include
Koilocytotic cells, exhibiting enlarged nuclei with abnormal contours and often associated with HPV
infection, and Dyskeratotic cells, characterized by hasty keratinization within cells or clusters, indicative
of HPV infection even in the absence of koilocytes. Metaplastic cells, displaying large
or small squamous cells with distinct cellular borders and vacuoles, are linked to higher detection
rates of pre-cancerous lesions in Pap tests. This dataset offers valuable insights into cervical cell
morphology, aiding in the identification and classification of both normal and abnormal cells in Pap
smears.
5.2 DATA PRE-PROCESSING
Before training the model for cervical cancer detection using Pap smear images, it’s crucial to preprocess
the dataset and split it appropriately. The dataset was divided into three sets: 70% for
training, 15% for validation, and 15% for testing, ensuring a balanced distribution of samples across
classes in each set to avoid bias.
After splitting the data, normalization techniques were applied. This included mean subtraction,
where the mean value of each channel was subtracted from the input image, and standardization,
which scales the feature value to have a mean of 0 and standard deviation of 1. These techniques
5

ensure consistent model performance across different data scales and prevent the doinance of
features with larger magnitudes.
Furthermore, data augmentation techniques such as flipping, rotating, scaling, and adding noise
were employed to increase the diversity of the training dataset, enhancing the model’s ability to
generalize. It’s essential to maintain a balanced distribution of samples across classes in each set
to avoid bias, thereby ensuring the model’s effectiveness in detecting both normal and abnormal
cervical cells.
5.3 TRAINING & VALIDATION OF THE MODEL
The process of training and validating the proposed model for cervical cancer cell classification involves
several crucial steps aimed at optimizing performance and ensuring generalization to unseen
data. Initially, during the forward pass, input images traverse through the ResNet50-Transformer
model, generating initial predictions that serve as the foundation for the model’s learning process.
Subsequently, the model’s performance is assessed using a loss function, quantifying disparities
between predicted and actual labels through loss computation. This iterative process guides the
backward pass or back propagation, where the loss is retroactively propagated through the network,
determining each parameter’s contribution to the overall loss. Optimization algorithms like Adam
adjust model parameters, iteratively minimizing loss and refining prediction accuracy.
Moreover, periodic validation steps on separate datasets are critical for monitoring the model’s generalization
and preventing over fitting. By assessing the model’s performance on unseen data, researchers
can identify potential issues with over fitting and fine-tune model parameters accordingly.
Following convergence or a predefined number of epochs, the trained model is evaluated on a test
dataset to gauge its efficacy in detecting cervical cancer. Metrics such as accuracy, precision, recall,
and confusion matrices are computed to quantify the model’s performance comprehensively. Utilizing
these iterative steps, researchers can develop and refine models adept at classifying cervical
cancer cells in Pap smear images, thereby advancing early detection efforts and enhancing patient
outcomes. Additionally, saving the model’s state dictionary at each epoch ensures reproducibility
and facilitates model comparison and optimization.
5.4 FINE-TUNING AND OPTIMIZATION
Fine-tuning and optimization are pivotal for maximizing the performance of the cervical cancer cell
classification model. Delving into hyper parameters like learning rate, batch size, and optimizer
selection is crucial as they profoundly impact the model’s convergence rate, stability, and overall
effectiveness. The choice of the optimizer, such as Adam, coupled with the utilization of the
categorical cross entropy loss function and soft max activation, sets the groundwork for efficient
training dynamics. Defining a batch size of 128 and training for 100 epochs provides initial parameters,
which can be fine-tuned to achieve optimal results. Experimentation with various combinations
allows for the identification of configurations enabling the model to adeptly learn relevant features
from the data.
Moreover, adjusting the model architecture based on performance feedback is vital. Tweaking architectural
parameters like layer count, filter sizes, and activation functions optimizes the model’s
capability to capture pertinent features in Pap smear images, enhancing its accuracy in cervical cell
abnormality classification. These hyper parameters and architectural choices hold paramount importance
as they ensure effective learning, generalization to unseen data, and prevention of over fitting.
By refining the model’s architecture and hyper parameters, researchers can develop a robust system
adept at accurately classifying cervical cell abnormalities, thus contributing to early detection efforts
and improving patient outcomes in cervical cancer diagnosis.
5.5 TESTING THE DATA ON TEST DATASET
Once the model has undergone training and validation, it is essential to evaluate its performance on
unseen data, which is typically done using a separate test dataset. This evaluation involves computing
various performance metrics such as accuracy, precision, recall and F1-score. These metrics
provide quantitative measures of the model’s ability to correctly classify normal and abnormal cervical
cells, offering insights into its overall performance.
6

Moreover, in addition to numerical metrics, analyzing the confusion matrix offers valuable insights
into the model’s classification performance across different classes. By examining the distribution
of true positive, true negative, false positive, and false negative predictions, healthcare professionals
can gain a deeper understanding of the model’s strengths and weaknesses. This analysis helps identify
specific areas where the model may struggle to distinguish between different classes, guiding
further improvements or adjustments.


6 RESULTS
Figure 4: Model Accuracy Comparison
The ResNet50-Transformer model presents a compelling solution for cervical cancer diagnosis, as
evidenced by its impressive performance metrics and robustness across various evaluation criteria.
With a training accuracy of 96.8% and a validation accuracy of 94.3%, the model demonstrates
strong learning capabilities and generalization to unseen data, essential for reliable diagnostic systems.
Notably, its resilience against overfitting is evident through consistent performance between
training and validation sets, ensuring reliable predictions in real-world scenarios. The achieved test
accuracy of 92.85% further validates its efficacy, affirming its potential for deployment in clinical
settings to facilitate early intervention and improve patient outcomes.
Figure 5: Performance Metrics
Analyzing the precision, recall, and F1-scores across different classes, the model exhibits high precision,
particularly notable in the Parabasal class, where a perfect precision score of 1.00 indicates
an absence of false positives. Furthermore, the model demonstrates remarkable recall values for
Parabasal and Superficial-Intermediate classes, indicating its ability to identify almost all relevant
cases within these categories. The balance between precision and recall, as reflected in the F1-scores,
further underscores the model’s robust performance across diverse cellular anomalies, contributing
to its reliability in accurate classification.
In assessing the confusion matrix, the model excels in accurately predicting Dyskeratotic and
Koilocytotic classes with minimal misclassifications, although some confusion persists between
Koilocytotic and Metaplastic classes. Conversely, predictions for Parabasal and Superficial-
Intermediate classes are near flawless, highlighting the model’s proficiency in identifying specific
cellular anomalies crucial for diagnostic purposes.
Overall, the ResNet50-Transformer model showcases strong performance across all evaluated metrics
and classes, with exceptional accuracy observed in certain categories. Despite minor confusion
between certain classes, its impact remains minimal on overall performance, underscoring the
7

Figure 6: Confusion Matrix of test data
model’s robustness for classifying cellular anomalies accurately. This robust performance makes it
a suitable candidate for practical applications where precise identification of these anomalies is imperative,
thereby contributing to advancements in healthcare diagnostics and ultimately improving
patient outcomes.


7 DISCUSSION
The model’s performance demonstrates its efficacy in cervical cancer cell classification, achieving
a commendable accuracy of 91% on the test dataset. Furthermore, the low loss value indicates the
model’s ability to minimize prediction errors effectively. The negligible difference between validation
and test accuracy suggests that the model generalizes well to unseen data, indicating resilience
against overfitting. Despite its computational intensity leading to prolonged inference times, the
model’s reliability remains high, as evidenced by the confusion matrix and comprehensive model
evaluation. The consistency across various evaluation metrics underscores the model’s robustness
and suitability for practical deployment in clinical settings. These results highlight the potential of
advanced deep learning techniques, such as ResNet50 and transformer architectures, in enhancing
cervical cancer screening and diagnosis, contributing to more effective healthcare solutions. However,
to address the issues raised by long inference times and guarantee that the model is more widely
accessible and scalable for practical uses, it is necessary to maximize computing efficiency.


8 LIMITATIONS
The ResNet50-Transformer model for cervical cancer diagnosis presents promising advancements
in healthcare, yet several limitations must be addressed to maximize its impact. Foremost among
these is the challenge of prolonged inference time, which could impede its real-time applicability
in clinical settings and disrupt workflow efficiency. Additionally, the complexity of the model
architecture may hinder its scalability and deployment, necessitating significant computational resources
for inference and potentially limiting widespread adoption. Despite efforts to enhance the
model’s generalization capabilities, challenges may persist in accurately diagnosing cervical cancer
across diverse patient populations, particularly if the training data lacks representation of all
demographic and clinical variations. Lastly, rigorous clinical validation is imperative to assess the
model’s performance and reliability in real-world healthcare settings, a process that entails regulatory
approval, validation studies, and integration into existing clinical workflows, which can be
both time-consuming and resource-intensive. Addressing these limitations is essential to realizing
the full potential of the ResNet50-Transformer model in improving healthcare outcomes and disease
management strategies.


9 FUTURE WORK & SIGNIFICANCE
Early detection of cervical cancer greatly increases the likelihood of successful treatment and improved
survival rates, underscoring the critical importance of reliable diagnostic tools in combating
8

this disease.However, the deployment of the ResNet50-Transformer model for cervical cancer diagnosis
is not without its challenges. One notable obstacle encountered during model execution is the
prolonged inference time, which can impede real-time diagnosis and workflow efficiency. Addressing
this challenge is paramount to the practical implementation of the model in clinical settings.
To mitigate the issue of prolonged inference time, several strategies can be explored. Model pruning
offers a promising approach by identifying and eliminating less critical parameters from the
model, effectively reducing its size without compromising performance. By streamlining the model
architecture, inference time can be significantly accelerated, enabling rapid analysis of patient data.
Quantization presents another viable solution for enhancing inference speed. By reducing the precision
of weights and activations within the model, quantization effectively reduces memory usage
and computational complexity, leading to faster inference times without sacrificing accuracy.
Moreover, the adoption of automated hyperparameter tuning techniques can further optimize model
performance and convergence speed. By systematically exploring a diverse range of hyperparameters,
these methods facilitate the discovery of optimal configurations, thereby enhancing the model’s
predictive capabilities and efficiency.
Additionally, expanding the model’s training dataset to encompass a larger and more diverse range
of cervical cancer cases holds promise for improving diagnostic accuracy and reliability. Access to
comprehensive datasets facilitates robust model training, enabling the model to learn from a broader
spectrum of pathological variations and clinical scenarios. This, in turn, enhances the model’s generalization
capabilities and ensures its effectiveness across diverse patient populations.
In conclusion, the ResNet50-Transformer model offers an advanced tool for cervical cancer diagnosis,
with the potential to significantly enhance patient care and outcomes. By addressing challenges
such as inference time and leveraging strategies for optimization and dataset expansion, the model
can be further refined to deliver accurate, timely, and accessible diagnostic solutions, ultimately
contributing to improved healthcare outcomes and disease management strategies.


REFERENCES

SIPAKMeD. URL https://www.cs.uoi.gr/˜marina/sipakmed.html.

O. Attallah. Cercan·net: Cervical cancer classification model via multi-layer feature ensembles of
lightweight cnns and transfer learning. Expert Syst. Appl., 229(PB), nov 2023. ISSN 0957-4174.
do i: 10.1016/j.eswa.2023.120624. URL https://doi.org/10.1016/j.eswa.2023.
120624.

J. S. Mandelblatt, W. F. Lawrence, L. Gaffikin, K. K. Limpahayom, P. Lumbiganon, S. Warakamin,J. King, B. Yi, P. Ringers, and P. D. Blumenthal. Costs and Benefits of Different Strategies to
Screen for Cervical Cancer in Less-Developed Countries. JNCI: Journal of the National Cancer
Institute, 94(19):1469–1483, 10 2002. ISSN 0027-8874. doi: 10.1093/jnci/94.19.1469. URL
https://doi.org/10.1093/jnci/94.19.1469.

I. Pacal and S. Kılıcarslan. Deep learning-based approaches for robust classification of cervical
cancer. Neural Comput. Appl., 35(25):18813–18828, jul 2023. ISSN 0941-0643. doi: 10.1007/
s00521-023-08757-w. URL https://doi.org/10.1007/s00521-023-08757-w.

Y. R. Park, Y. J. Kim, W. Ju, K. H. Nam, S. Kim, and K. G. Kim. Comparison of machine and
deep learning for the classification of cervical cancer based on cervicography images. Scientific
reports, 11(1), 8 2021. doi: 10.1038/s41598-021-95748-3. URL https://doi.org/10.
1038/s41598-021-95748-3.

A. Tripathi, A. Arora, and A. Bhan. Classification of cervical cancer using deep learning algorithm.
In 2021 5th International Conference on Intelligent Computing and Control Systems (ICICCS),
pages 1210–1218, 2021. doi: 10.1109/ICICCS51141.2021.9432382.


S.-Y.Wang, O.Wang, R. Zhang, A. Owens, and A. A. Efros. Cnn-generated images are surprisingly
easy to spot... for now, 2020. 

WHO. Cervical cancer, 3 2024. URL https://www.who.int/news-room/
fact-sheets/detail/cervical-cancer.9

O. Yaman and T. Tuncer. Exemplar pyramid deep feature extraction based cervical cancer imageclassification model using pap-smear images. Biomedical Signal Processing and Control, 73:103428, 03 2022. doi: 10.1016/j.bspc.2021.103428.

C. Zhao, R. Shuai, L. Ma, W. Liu, and M. Wu. Improving cervical cancer classification with imbalanced datasets combining taming transformers with T2T-ViT. Multimedia tools and applications, 81(17):24265–24300, 3 2022. doi: 10.1007/s11042-022-12670-0. URL https: //doi.org/10.1007/s11042-022-12670-0.10
