Example codes and experiments around generative models using TensorFlow, Keras, Sklearn...

Generative models are type of unsupervised learning to generate new sample outputs by recognizing patterns from inputs. 
The applications of generative models are: next video frame prediction, text to image generation, image to image translation, enhancing image resolution, colorization, synthesize a face animated by a person’s movement, reinforcement learning, generate realistic samples of images, etc.
They categorized in two main categories: a) explicit models which assume some prior distribution about the data, such as Kernel Density Estimation (KDE), and b) impicit models which define a stochastic procedure that directly generates data, such as Generative Adversarial Networks (GANs).


- Gaussian Mixture Models (GMM): GMM is a probabilistic model that assumes all data points are generated from a mixture of a finite number of Gaussian distribution with unknown parameters.
In other words, GMM is a combination of weighted Gaussians (function composed of several Gaussians), where each mixture component specified by:
μ: a mean that defines center of each Gaussian distribution,
σ2 / Σ: a variances/covariances matrix that define its width, and
π: a mixing probability that defines how big or small the Gaussian function will be (sum to 1).


- Generative Adversarial Networks (GANs):
Generative Adversarial Networks are deep learning based generative models, while they create new data instances that resemble training data.
GAN is a system where two competing neural networks compete with each other to create variations in the data: a) generator, which takes a sample and generates a new sample of data, and b) discriminator, which decides whether data is generated or taken from the real sample or not.
In typical, the generator tries to fool the discriminator, and the discriminator tries to keep from being fooled.


- Variational Autoencoders (VAE): Variational autoencoders map input on to a distribution, instead of mapping input to a fixed vector in latent bottleneck. 
It means that the normal bottleneck is replaced by two vectors: mean and standard deviation.


- 
