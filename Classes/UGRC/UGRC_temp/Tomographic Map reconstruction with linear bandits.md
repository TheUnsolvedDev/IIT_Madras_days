# 1 Introduction
Tomography is a technique where imaging is done by sectioning the object with some penetrating wave. The uses of this method are mostly used in archaeology, radiology, material science biology etc. Various types of tomography include Aerial tomography with Electromagnetic radiation, Electron tomography done by transmission electron microscopy, Functional magnetic resonance imaging and many more. In this part, we will focus on radio tomography where the radio waves will be used for reconstruction of the environment. The ultimate objective is to reconstruct a comprehensive and detailed map of the environment. Achieving this goal demands a nuanced approach to path planning, where linear bandits play a pivotal role in optimising the solution. The way it will be done is by sending radio waves from a particular point and receiving the radio waves at some other point noting the total attenuation and reconstruction of the whole surrounding through proper path planning by the use of linear bandits, by framing it to a bandit model to solve the linear inverse problem with minimal and efficient amount of actions and data needed for reconstructing the whole map. Linear bandits, a concept borrowed from the field of artificial intelligence, are instrumental in framing the radio tomography problem into a bandit model. This approach allows for the efficient resolution of the linear inverse problem, minimising the number of actions and data required to reconstruct the entire map.

# 2 Background and Literature Review
Use of this strategy arises when we need to know about a particular environment that is hard to explore while venturing deep inside so we can also implement this strategy of exploration to get aware of the surroundings. Also, there is one advantage as sometimes for some exploration vehicles a concerning fact arises when it has a fixed line of sight that it needs to explore but this strategy is beneficial in such scenarios as the Radio frequency can penetrate through the object and we can reconstruct a piece of partial information regarding the object for our necessities.

A linear inverse problem refers to a type of mathematical problem where the goal is to determine the input of a linear system given the output, taking into account the uncertainties and noise in the measurements. In other words, it involves finding the cause or input that resulted in a known effect or output through a linear relationship. $y = Ax + \epsilon$. Here we are solving this to derive the map from the set of actions and the resultant attenuation values. 
## 2.1 Background
The use case of this is quite dominant in the practical aspects of imaging systems. First, radar systems transmit RF probes and receive the signals caused by the objects in an environment. A time delay between the transmission and reception describes the distance to the sender. Second, The measurements along the paths are used to compute an estimate of the spatial field of the transmission parameters throughout the medium which is done in the cases of computed tomography (CT) methods in medical and geophysical imaging systems that use signal measurements along many different paths through a medium. 
## 2.2 Literature Review

# 3 Program and Design of the Problem and the solution
As for the Problem Design mostly under current circumstances, it is done under the simulation where an Environment is designed that will make a 2 Dimensional Map \autoref{fig:maps} of a given size and a particular sparsity will be generated terming it the real map of the environment, from this, action is passed this action value is described as the location of the sender and the receiver this gives us the attenuated result of the signal received by the sender after the sender sends it. The Bresenham line is drawn from the sender and receiver denoting as a binary mask to describe the points that are to be taken as the action vector of our environment, denoted as the action taken at time t, $a_t$ and this vector is flattened. Then the projection of this with the original map flattened gives us the resultant attenuation describing the value as $b_t$. This gives us a linear relationship with the action and the resultant attenuation in the case of our simulation. Under the given diagram \autoref{fig:bresenham} show how the Bresenham line is used to calculate the action vector, all the green parts are described as $1$ and the rest are $0$. The $(x_1,y_1)$ is described as the location of the sender and $(x_2,y_2)$ is described as the location of the receiver.
## 3.1 Objectives
The objective is to reconstruct the original map with minimal error from the original one (in real case scenario the original one is not known to us, that is we need to construct it from the attenuation values and the actions we chose) by choosing fewer amounts of action to get the desired result. Taking a huge amount of actions will cost the sender and the receiver as they need to travel more distance. This objective leads to an important strategy for exploration and exploitation strategy for choosing the action to get the best-estimated map. The need for solving this linear inverse problem is to make it efficient enough to get it implemented in real-time systems.
## 3.2 Methodology

# 4 Results
# 5 Ongoing Work

# References