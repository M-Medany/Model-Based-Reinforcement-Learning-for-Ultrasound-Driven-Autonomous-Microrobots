# Model-Based-Reinforcement-Learning-for-Ultrasound-Driven-Autonomous-Microrobots# Model-Based Reinforcement Learning for Ultrasound-Driven Autonomous Microrobots

## Abstract
AI has catalyzed transformative advancements across multiple sectors, from medical diagnostics to autonomous vehicles, enhancing precision and efficiency. As it ventures into microrobotics, AI offers innovative solutions to the formidable challenge of controlling microrobots, which typically operate within imprecise, remotely actuated systems. We implement state-of-the-art model-based reinforcement learning for autonomous control of an ultrasound-driven microrobot learning from recurrent imagined environments. Our non-invasive, AI-controlled microrobot offers precise propulsion, which efficiently learns from images in data-scarce environments. Transitioning from a pre-trained simulation environment, we achieve sample-efficient collision avoidance and channel navigation, reaching a **90% success rate** in target navigation across various channels within an hour of fine-tuning. Moreover, our model initially successfully generalized in **50% of tasks in new environments**, improving to **over 90% with 30 minutes of further training**. Furthermore, we have showcased real-time manipulation of microrobots within complex vasculatures and across stationary and physiological flows, underscoring AI's potential to revolutionize microrobotics in biomedical applications.

This repository provides code for both simulation environments and physical experiments. The simulation environment can be run on Linux-based systems or within a virtualized Ubuntu environment on Windows using WSL2.

For real-world experiments, the setup requires integrating the code with an imaging system, a Tektronix function generator, and an electronic circuit for switching between piezoelectric transducers (PZTs). Additionally, a camera must be connected to enable real-time feedback for microrobot tracking and manipulation.

![Figure 1](results/Figure%201a.png)
### **Figure 1**
Schematic of the experimental setup, showcasing an artificial vascular channel with eight PZTs in an octagonal configuration (left image). A schematic illustrates the microrobot’s behavior under ultrasound activation and details methods for its manipulation (right image).


![Figure 2](results/Figure%201F.png)
### **Figure 2**
The microrobot envisions multiple potential paths towards a target in a dreamed environment, allowing it to 'dream' and train on various possible scenarios simultaneously. a recurrent 'dream' in the latent space 

## Demonstration GIFs
Below are key demonstrations from our research:

### **Movie S1**
Training process of MBRL in a real microfluidic racetrack channel. The left video shows early training (<100k steps) where the algorithm struggles to reach the target. On the right, after 340k steps, the algorithm shows improved target acquisition. The full process spans 10 days due to real-environment interactions.

![Movie S1](results/Movie%20S1.gif)

### **Movie S2**
Transfer learning behavior from a simulation environment to a real experimental environment of the same shape. It shows that the model converges in real experiments in just 3 hours.

![Movie S2](results/Movie%20S2.gif)

### **Movie S3**
Continuous action training in a vascular channel. The left side shows RRT* blue tree branches searching for the shortest path, marked in red when found. On the right, the microrobot is marked in blue, with the next target in red. The video demonstrates microrobots attempting to follow the path in real time.

![Movie S3](results/Movie%20S3.gif)

### **Movie S4**
Transfer learning from a simulation environment to a real vascular channel using an MBRL model with sweeping actions.

![Movie S4](results/Movie%20S4.gif)

### **Movie S5**
MBRL general model trained on 10 environments demonstrates its ability to perform across all 10 environments and adapt to a new, unseen channel with just 30 minutes of additional training.

![Movie S5](results/Movie%20S5.gif)

### **Movie S6**
Autonomous manipulation in a flow environment after transfer learning from a simulation that mimics the flow, guiding the microrobot to move in a low-drag region near the wall.

![Movie S6](results/Movie%20S6.gif)

### **Movie S7**
Active and passive shape-shifting of a microrobot navigating obstacles in a microchannel is demonstrated. Passive deformation occurs when a single piezoelectric transducer (PZT) is activated, while active manipulation involves dynamic shape changes using multiple PZTs for precise control and navigation.

![Movie S7](results/Movie%20S7.gif)


## Credits & Dependencies  
This project utilizes and adapts components from the following repositories:  

- **DreamerV3** – A model-based RL algorithm used for microrobot control.  
GitHub: [https://github.com/danijar/dreamerv3](https://github.com/danijar/dreamerv3)  

- **Segment Anything Model (SAM)** – A segmentation model used for microrobot detection and tracking.  
  GitHub: [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)  

- **RSA_API** – Integrated for function generator control.  
  GitHub: [https://github.com/tektronix/RSA_API](https://github.com/tektronix/RSA_API)  

