# project C5.  A system with memory bank for vision model (Self-grow)

## Version 2
After discussing with Mirela, I had something more solid:
This is about how vision system works in us as human being, by mimicking us, we had some good insight about it. Based on the reference, here is the AI version of it:
1. A model encoder E pretrained so it can extract image features. For the first cat image I_cat, we use E(I_cat) to extract its features, and associate them with a label ‘cat’ 
2. We need to create a memory dictionary M = {} that store the representation of cat images E(I_cat) and store them locally - M[‘cat’] = E(I_cat)
3. More cat images come in and stabilise the M[‘cat’], need a mechanism to do this, and slight different, instead of RL, we can simple use a SL is fine
4. A new image comes in, there is a feature matching algorithm to decide if it is cat or not. If it is cat, then go to step 3; if not, go to step 6
5. Inference is also related to the image background
6. Feature matching algorithm is needed to match or differentiate different samples

Here is the algorithm in making this AI architecture to mimic how vision works
1. Pretrain a model E that can extract vision features, frozen E’s weights
2. Forward image I to E for the feature extraction R = E(I)
3. Associate R with a label L to form the memory M = {L: R}, 
4. perform step3 N times (N is the number of classes) and repeat K times (so each label can have K samples), update M_stable by average the Rs for each L. R_stable = (R_1+R_2+..+R_K)/K [optional but important, might be good to use contrastive learning technique to make each class samples’s representation far away]
5. Take in new samples and match. Use d_i ,the distance between R_new and R_stable for each classes I to get the D = [1/d_1, 1/d_2, …, 1/d_N], and normalise this to get the results of which label it should be and its probability.
PS: E might be not enough for the model to separate each labels nicely, so a new trainable layer(s) might need to involve in between in order to update the cognition.