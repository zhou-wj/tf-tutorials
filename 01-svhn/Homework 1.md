# Homework 1

## Requirement
- The work has to be finished individually. Plagiarism will be dealt with seriously.

## Goals
- Learn to do control experiments
- Learn there are alternatives to Softmax/Cross Entropy when training DNN

## FAQ
Q: Where can I get support for this homework?
A: Use "Issues" of this repo.

Q: What's the DDL for homeworks?
A: We'll discuss the homework in succeeding experiment course (every two week). Those homeworks turned in after discussion will be capped at 90 marks.

Q: How will the score of each homework affect final course score?
A: The algorithm is TBD.

## GPU servers
- We provide some servers, each server has eight Nvidia GPU for students to do all the experiments.
- **python** and **tensorflow** has already been installed.
- **~/course/dataset** has all the data for each assignment. For example **~/course/datasest/SVHN/** contains train and test data for homework1.
- **~/course/tutorial** has the start code we provide for each assignment. Start code for SVHN is in **~/course/tutorial/01-SVHN**
- **~/students** is used for writing your own code. 
        - First build a directory under **~/students/** using your own name, e.g. **~/students/zhaoyuekai**
        - Then copy the start code we provide to your own directory.
        - **Do not make any change to files under ~/course directory!!**

## Warnings
- **Do not use the computation resource for your private project!!**
- **Do not let anyone who has not selected this course know the username and password of our servers!!**

## Questions
- #### Q1: Finding alternatives of softmax

  <img src="./images/find_soft.png" width="500px"/>

- #### Q2: Regression vs Classification

  - Change cross entropy loss to the square of euclidean distance between model predicted probability and one hot vector of the true label.

- #### Q3: Lp pooling

  - Change all pooling layers to Lp pooling
  - Descriptions about Lp pooling is at https://www.computer.org/csdl/proceedings/icpr/2012/2216/00/06460867.pdf

- #### Q4: Regularization

  - Try Lp regularization with different p.
  - Set Lp regularization to a minus number. (L_model + L_reg to L_model - L_reg)

- #### Q5: Where to find dataset files?
  - Open http://ufldl.stanford.edu/housenumbers
  - Please down format2. (train\_32x32.mat, test\_32x32.mat, extra\_32x32.mat)
