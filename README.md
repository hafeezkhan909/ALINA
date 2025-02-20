# ALINA: Advanced Line Identification and Notation Algorithm

<p align="justify"> The usage of machine learning and deep learning for designing and deploying cutting-edge systems is exponentially rising, and vast volumes of data with suitable labels are necessary to create these models. In this research, the focus is on the development of autonomous aircraft perception system for accurately detecting and labeling the line markings on an airport taxiway. A generalized framework called an Automated Line Identification and Notation Algorithm (ALINA) has been proposed, which can be used for labeling any taxiway datasets that may consist of complex environmental settings, changeable weather, and atmospheric conditions. ALINA begins with an interactive process for specifying a trapezoidal region of interest (ROI), which is subsequently used across all images in the dataset. The ROI undergoes geometrical modification and color space transformation for generating a binary map of the pixels, which is then subjected to histogram analysis. A CIRCular threshoLd pixEl Discovery And Traversal (CIRCLEDAT) algorithm has been proposed as part of ALINA, which acts as an integral step in determining the pixels corresponding to line markings on the airport taxiway. The output generates annotations of the line markings on the image along with a text file of its corresponding coordinates. Using this approach, 60,249 images from the taxiway dataset have been labeled accurately. The detection rate obtained by ALINA after testing it with 120 ground truth images consisting of different scenarios of the airport taxiway was recorded as 0.9844, attesting its dependability and effectiveness. </p>

Link to paper: https://openaccess.thecvf.com/content/CVPR2024W/VDU/papers/Khan_ALINA_Advanced_Line_Identification_and_Notation_Algorithm_CVPRW_2024_paper.pdf
