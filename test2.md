Antigen-Specific Antibody Design and Optimization
with Diffusion-Based Generative Models
for Protein Structures
Shitong Luo1∗ , Yufeng Su2∗ , Xingang Peng3 , Sheng Wang4 , Jian Peng1,2 , Jianzhu Ma1,5
1
Helixon Research
2
University of Illinois Urbana-Champaign
3
School of Intelligence Science and Technology, Peking University
4
Paul G. Allen School of Computer Science, University of Washington
5
Institute for AI Industry Research, Tsinghua University
luost@helixon.com,luost26@gmail.com
swang@cs.washington.edu,jianpeng@illinois.edu,majianzhu@tsinghua.edu.cn

Abstract
Antibodies are immune system proteins that protect the host by binding to specific
antigens such as viruses and bacteria. The binding between antibodies and antigens
is mainly determined by the complementarity-determining regions (CDR) of the
antibodies. In this work, we develop a deep generative model that jointly models
sequences and structures of CDRs based on diffusion probabilistic models and
equivariant neural networks. Our method is the first deep learning-based method
that generates antibodies explicitly targeting specific antigen structures and is one
of the earliest diffusion probabilistic models for protein structures. The model is
a “Swiss Army Knife” capable of sequence-structure co-design, sequence design
for given backbone structures, and antibody optimization. We conduct extensive
experiments to evaluate the quality of both sequences and structures of designed
antibodies. We find that our model could yield competitive results in binding
affinity measured by biophysical energy functions and other protein design metrics.

1

Introduction

Antibodies are important immune proteins generated during an immune response to recognize and
neutralize the pathogen [Janeway et al., 2001]. As illustrated in Figure 1a, an antibody contains two
heavy chains and two light chains, and their overall structure is similar. Six variable regions determine
the specificity of an antibody to the antigens. They are called the Complementarity Determining
Regions (CDRs), denoted as H1, H2, H3, L1, L2, and L3. Therefore, the most important step for
developing effective therapeutic antibodies is to design CDRs that bind to the specific antigen [Presta,
1992, Akbar et al., 2022a].
Similar to other protein design tasks, the search space of CDRs is vast. A CDR sequence with L amino
acids has up to 20L possible protein sequences. It is not feasible to test all the possible sequences
using experimental approaches, so computational methods are needed. Traditional computational
approaches rely on sampling protein sequences and structures from complex biophysical energy
functions [Pantazes and Maranas, 2010, Lapidoth et al., 2015, Adolf-Bryfogle et al., 2018, Warszawski
et al., 2019]. They are generally time-consuming and are prone to get trapped in local optima.
Recently, various deep generative models have been developed to design antibodies [Saka et al., 2021,
Akbar et al., 2022b, Jin et al., 2022]. Compared to conventional algorithms, deep generative models
∗

Equal contribution.

36th Conference on Neural Information Processing Systems (NeurIPS 2022).

(a)

(b)

Antigen

Heavy Chain
Light Chain

1

2

3

CDR-H

3

Same Cα Distance
Between Amino Acids

Antigen

1

2

R
N Cα
C

CDR-L

Cα C
R

H1
H2

H3

L3

CDR

L1

N

R Cα

L2

N

Antibody

C

Cα

N

Interaction

R

CDR

R
Cα C

N

C
N Cα
R

Interacting
N

N

Cα R
C

R Cα
C

Non-interacting

C
N
Cα
C

R

N
R Cα
C

N
Cα R
C

Figure 1: (a) Antibody-antigen complex structure and CDR structure. (b) The orientations of amino
acids (represented by triangles) determine their side-chain orientations, which are key to inter-aminoacid interactions.
could directly capture higher-order interactions among amino acids on antibodies and antigens and
generate antibodies more efficiently [Akbar et al., 2022a]. Recently, Jin et al. proposed a generative
model for antibody structure-sequence co-design. Their model addresses two important computational
challenges: First is how to model the intrinsic relation between CDR sequences and 3D structures,
and second is how to model the distribution of CDRs conditional on the rest of the antibody sequence.
However, there is still a large gap to fill before generative models become practical for antibody
design.
Here, we identify another three challenges for antibody sequence-structure co-design. First, the model
should be explicitly conditional on the 3D structures of the antigen and generate CDRs that fit the
antigen structure in the 3D space. This is indispensable for the model to generalize to new antigens.
Second, the interactions between amino acids are mainly determined by side-chains which are groups
of atoms stretching out from the protein backbone (Figure 1b) [Liljas et al., 2016]. Therefore, the
model should be able to consider both the position and orientation of amino acids. Third, in drug
discovery, pharmacologists collect multiple initial antibodies either from humanized mice or patients
[Presta, 1992, Barlow et al., 2018, Warszawski et al., 2019]. Therefore, instead of de novo design, the
model should be applicable to another realistic scenario: optimizing a particular antibody to increase
the binding affinity to the antigen. To the best of our knowledge, no previous machine learning model
satisfies all of the above design principles.
Generative Model

To address these challenges, we propose a diffusion-based
Antigen
Structure
generative model [Sohl-Dickstein et al., 2015, Song and
CDR
Designs
Ermon, 2019, Ho et al., 2020, Yang et al., 2022] capable
of jointly sampling antibody CDR sequences and struc?
tures. More importantly, the joint distribution of a CDR
ARALYYYDSSGYDAYYFDY
sequence and its structure is directly conditional on antiARGRRVSAPNAYGAREGLQ
ARGLDPLYCNNSTTCYRVG
Antibody
ARGHTLYGAVLMENYHYGM
gen structures. Given a protein complex consisting of an Framework
antigen and an antibody framework as input2 (as illustrated
in Figure 2), we first initialize the CDR with an arbitrary Figure 2: The task in this work is to desequence, positions, and orientations. The diffusion model sign CDRs for a given antigen structure
first aggregates information from the antigen and the an- and an antibody framework.
tibody framework. Then, it iteratively updates the amino
acid type, position, and orientation of each amino acid on CDRs. In the last step, we reconstruct
the CDR structure at the atom level using side-chain packing algorithms based on the predicted
orientations [Alford et al., 2017]. From the perspective of model capability, one of the most important
reasons for us to choose the diffusion-based model over other generative models such as generative
adversarial networks [Goodfellow et al., 2014] and variational auto-encoders [Kingma and Welling,
2013] is that it generates CDR candidates iteratively in the sequence-structure space so that we can
interfere and impose constraints on the sampling process to support a broader range of design tasks.
We summarize our contributions as follows:
• We propose the first deep learning models to perform antibody sequence-structure design by
considering the 3D structures of the antigen.
• In our model, we not only design protein sequences and coordinates but also side-chain
orientations (represented as SO(3) element) of each amino acid. It is the first deep learning
2

The structure of the antigen-antibody framework can be obtained either from existing antigen-antibody
structure or by docking an initial antibody to the target antigen.

2

model that could achieve atomic-resolution antibody design and is equivariant to rotation
and translation.
• We show that our model can be applied to a wide range of antibody design tasks, including
sequence-structure co-design, fix-backbone CDR design, and antibody optimization.

2

Related Work

Computational Antibody Design Conventional computational approaches are mainly based on
sampling algorithms over hand-crafted and statistical energy functions and iteratively modify protein
sequences and structures [Adolf-Bryfogle et al., 2018, Lapidoth et al., 2015, Warszawski et al., 2019,
Pantazes and Maranas, 2010, Ruffolo et al., 2021]. These methods are inefficient and prone to getting
stuck at local optima due to the rough energy landscape. In recent years, deep learning methods
have shown potential in antibody design by using language models to generate protein sequences
[Alley et al., 2019, Shin et al., 2021, Saka et al., 2021, Akbar et al., 2022b]. Although much more
efficient, the sequence-based methods can only generate new antibodies based on previously observed
antibodies but can hardly generate antibodies for specific antigen structures.
Jin et al. proposed the first CDR sequence-structure co-design deep generative model which focuses
on designing antibodies to neutralize SARS-CoV-2. It relies on an additional antigen-specific
predictor to predict the neutralization of the designed antibodies, which is not generalizable to
arbitrary antigens. In comparison to their model, we explicitly model the 3D structure of an
antigen, opening the door to generalizing the prediction to unseen antigens with solved 3D structures.
Another advantage of our model is that we consider not only backbone atom coordinates but also
the orientation of amino acids. The orientation is critical to protein-protein interactions as most of
the atoms interacting between antibodies and antigens are in the side-chain [Liljas et al., 2016] (as
illustrated in Figure 1b). Lastly, the model proposed by Jin et al. is not equivariant by construction,
which is fundamental in molecular modeling.
Protein Structure Prediction Protein structure prediction algorithms take protein sequences and
Multiple Sequence Alignments (MSAs) as input and translate them to 3D structures [Jumper et al.,
2021, Baek et al., 2021, Yang et al., 2020]. Accurate protein structure prediction models predict not
only the position of amino acids but also their orientation [Jumper et al., 2021, Yang et al., 2020].
The orientation of amino acids determines the direction in which its side chain stretches, so it is
indispensable for reconstructing full-atom structures. AlphaFold2 [Jumper et al., 2021] predicts
per-amino-acid orientations in an iterative fashion, similar to our proposed model. However, it is
not generative, unable to efficiently sample diverse structures for protein design. Recently, based on
prior protein structure prediction algorithms, methods for predicting antibody CDR structures have
emerged [Ruffolo et al., 2022b,a], but they are not able to design CDR sequences.
Diffusion-Based Generative Models Diffusion probabilistic models learn to generate data via
denoising samples from a prior distribution [Sohl-Dickstein et al., 2015, Song and Ermon, 2019,
Ho et al., 2020]. Recently, progress has been made in developing equivariant diffusion models
for molecular 3D structures [Shi et al., 2021, Hoogeboom et al., 2022, Jing et al., 2022, Xu et al.,
2022]. Atoms in a molecule do not have natural orientations, so the generation process differs from
generating protein structures. Diffusion models have also been extended to non-Euclidean data, such
as data in the Riemannian manifolds [Leach et al., 2022, De Bortoli et al., 2022]. These models are
relevant to modeling orientations which are represented by elements in SO(3). In addition, diffusion
models can also be used to generate discrete categorical data [Hoogeboom et al., 2021, Austin et al.,
2021]. Concurrently with this work, various diffusion probabilistic models have been developed for
proteins [Anand and Achim, 2022, Trippe et al., 2022, Wu et al., 2022].

3

Methods

This section is organized as follows: Section 3.1 introduces notations used throughout the paper and
formally states the problem. Section 3.2 formulates the diffusion process for modeling antibodies.
Section 3.3 introduces details about the neural network parameterization for the diffusion processes.
Section 3.4 presents sampling algorithms for various antibody design tasks.
3

3.1

Definitions and Notations

An amino acid in a protein complex can be represented by its type, Cα atom coordinate, and the
orientation, denoted as si ∈ {ACDEFGHIKLMNPQRSTVWY}, xi ∈ R3 , Oi ∈ SO(3), respectively. Here
i = 1 . . . N , and N is the number of amino acids in the protein complex3 .
In this work, we assume the antigen structure and the antibody framework is given (Figure 2), and we
focus on designing CDRs on the antibody framework. Assume the CDR to be generated has m amino
acids with index from l + 1 to l + m. They are denoted as R = {(sj , xj , Oj ) | j = l + 1, . . . , l + m}.
Formally, our goal is to jointly model the distribution of R given the structure of the antibody-antigen
complex C = {(si , xi , Oi ) | i ∈ {1 . . . N }\{l + 1, . . . , l + m}}.
3.2

Diffusion Processes

A diffusion probabilistic model defines two Markov chains of diffusion processes. The forward
diffusion process gradually adds noise to the data until the data distribution approximately reaches the
prior distribution. The generative diffusion process starts from the prior distribution and iteratively
transforms it to the desired distribution. Training the model relies on the forward diffusion process to
simulate the noisy data. Let stj , xtj , Otj denote the intermediate state of amino acid j at time step t.
Rt = {stj , xtj , Otj }l+m
j=l+1 represents the sequence and structure sampled at step t. t = 0 represents
the state of real data (observed sequences and structures of CDRs) and t = T represents samples from
the prior distribution. Forward diffusion goes from t = 0 to T , and generative diffusion proceeds in
the opposite way. The diffusion processes for amino acid types stj , coordinates xtj , and orientations
Otj are defined as follows:
Multinomial Diffusion for Amino Acid Types The forward diffusion process for amino acid types
is based on the multinomial distribution defined as follows [Hoogeboom et al., 2021]:


1
t−1
t
t
q(stj |st−1
)
=
Multinomial
(1
−
β
)
·
onehot(s
)
+
β
·
·
1
,
(1)
type
type
j
j
20
where onehot represents a function that converts amino acid type to a 20-dimensional one-hot vector
t
and 1 is an all-one vector. βtype
is the probability of resampling another amino acid over 20 types
t
uniformly. When t → T , βtype is set close to 1 and the distribution is closer to the uniform distribution.
The following probability density provides an efficient way to perturb s0j for timestep t during training
[Hoogeboom et al., 2021]:


1
t
t
q(stj |s0j ) = Multinomial ᾱtype
· onehot(s0j ) + (1 − ᾱtype
)·
·1 ,
(2)
20
Qt
t
τ
where ᾱtype
= τ =1 (1 − βtype
).
The generative diffusion process is defined as:


t
t
p(st−1
j |R , C) = Multinomial F (R , C)[j] ,

(3)

where F (·)[j] is a neural network model taking the structure context (antigen and antibody framework)
and the CDR state from the previous step as input and predicts the probability of the amino acid
type for the j-th amino acid on the CDR. Note that, different from the forward diffusion process, the
generative diffusion process must rely on the structure context C and the CDR state of the previous
step including positions and orientations. The main difference between these two processes is that the
forward diffusion process adds noise to data so it is irrelevant to data or contexts but the generative
diffusion process depends on the given condition and full observation of the previous step. The
t 0
generative diffusion process needs to approximate the posterior q(st−1
j |sj , sj ) derived from Eq.1 and
Eq.2 to denoise. Therefore, the objective of training the generative diffusion process for amino acid
types is to minimize the expected KL divergence between Eq.3 and the posterior distribution:
 X


1
t−1 t 0
t−1
t
t
DKL q(sj |sj , sj ) p(sj |R , C) .
(4)
Ltype = ERt ∼p
j
m
3

Note that a protein complex contains more than one chain, so N is not the length of one protein but is the
sum of the lengths of all chains in the complex.

4

<latexit sha1_base64="Y4hkdKshWOS+NgTZWSTwbUFJbvw=">AAACiXicbVHdTtswGHUyxiBsEOByNxYdEkisSiYEaFdFlSbugGkFpKZUjuu0pnYS2V8Qlclz8Ch7iD0B1zwD9zhphcbPJ1k+Pt/5fnQc54JrCIJ7x/0w93H+08Kit/T5y/KKv7p2prNCUdahmcjURUw0EzxlHeAg2EWuGJGxYOfxuF3lz6+Z0jxL/8AkZz1JhilPOCVgqb5f5pFgCWx59RUZHEkCIyWNLvtXlwa+h+XOlIsTc/MOd/zM4Ujx4Qii0otiPrzFXi2hRJjf5SXMKqpn2ypq6XbfbwTNoA78FoQz0Gj5/37t3v09POn7j9Ego4VkKVBBtO6GQQ49QxRwKpjtW2iWEzomQ9a1MCWS6Z6pbSrxpmUGOMmUPSngmv2/whCp9UTGVlmtql/nKvK9XLeA5KBneJoXwFI6HZQUAkOGK8/xgCtGQUwsIFRxuyumI6IIBfsznreJaaEhk3jW7+XgWJae9Sl87cpbcPajGe41d0/DRusbmsYC+oo20BYK0T5qoSN0gjqIogdn0Vlz1t0lN3QP3J9TqevMatbRi3DbTyzEyC4=</latexit>

<latexit sha1_base64="x4kd4LMZURiXD9YMoYXb8/W3tLY=">AAACF3icbVC7TsMwFHV4lvAqMLJYlEpMVYIqYKzEwlgQfUhtqBzXba3aSbBvkKoo34HY4EvYECsjH8KO02agLUeydHTOvb5Hx48E1+A439bK6tr6xmZhy97e2d3bLx4cNnUYK8oaNBShavtEM8ED1gAOgrUjxYj0BWv54+vMbz0xpXkY3MMkYp4kw4APOCVgJK8rCYwoEcld+gC9YsmpOFPgZeLmpIRy1HvFn24/pLFkAVBBtO64TgReQhRwKlhqd2PNIkLHZMg6hgZEMu0l09ApLhuljwehMi8APFX/biREaj2RvpnMQupFLxP/8zoxDK68hAdRDCygs0ODWGAIcdYA7nPFKIiJIYQqbrJiOiKKUDA92XYZ01hDKHH+3/xhX6a26cldbGWZNM8r7kWlelst1U7zxgroGJ2gM+SiS1RDN6iOGoiiR/SMXtGb9WK9Wx/W52x0xcp3jtAcrK9fCjifpA==</latexit>

Rt

20 Types

sTj

Gaussian
<latexit sha1_base64="iY+iHeKHqGFW5aH1g2zxqNaKIOQ=">AAACHnicbVDLSsNAFJ3UV42vWJduBkuhq5JIUZcFNy4r9AVtLJPppB07k4SZibSE/Iq40y9xJ271Q9w7abOwrQcGDufcO/dwvIhRqWz72yhsbe/s7hX3zYPDo+MT67TUkWEsMGnjkIWi5yFJGA1IW1HFSC8SBHGPka43vc387hMRkoZBS80j4nI0DqhPMVJaGlqlAUdq4vnJLH1IWukweUyHVtmu2QvATeLkpAxyNIfWz2AU4piTQGGGpOw7dqTcBAlFMSOpOYgliRCeojHpaxogTqSbLLKnsKKVEfRDoV+g4EL9u5EgLuWce3oySyrXvUz8z+vHyr9xExpEsSIBXh7yYwZVCLMi4IgKghWba4KwoDorxBMkEFa6LtOsQBxLFXKY/7d62OOpqXty1lvZJJ3LmnNVq9/Xy41q3lgRnIMLUAUOuAYNcAeaoA0wmIFn8ArejBfj3fgwPpejBSPfOQMrML5+AT+0omI=</latexit>

SO(3) Unif.
<latexit sha1_base64="djRA4VDVKrmYm4fKEAecKBlmLOU=">AAACHnicbVDLSsNAFJ3UV42vWJduBkuhq5JIUZcFN+6s0Be0NUymk3bsTBJmJmIJ+RVxp1/iTtzqh7h30mZhWw8MHM65d+7heBGjUtn2t1HY2Nza3inumnv7B4dH1nGpI8NYYNLGIQtFz0OSMBqQtqKKkV4kCOIeI11vep353UciJA2DlppFZMjROKA+xUhpybVKA47UxPOT2/Q+aaVu8pC6Vtmu2XPAdeLkpAxyNF3rZzAKccxJoDBDUvYdO1LDBAlFMSOpOYgliRCeojHpaxogTuQwmWdPYUUrI+iHQr9Awbn6dyNBXMoZ9/RkllSuepn4n9ePlX81TGgQxYoEeHHIjxlUIcyKgCMqCFZspgnCguqsEE+QQFjpukyzAnEsVchh/t/yYY+npu7JWW1lnXTOa85FrX5XLzeqeWNFcArOQBU44BI0wA1ogjbA4Ak8g1fwZrwY78aH8bkYLRj5zglYgvH1C/p1ojk=</latexit>

xTj

OTj

<latexit sha1_base64="33DRJKG18MS8OXoW0d6iDNVX96s=">AAACHnicbVDLTgIxFO3gC8cX4tJNIyFhRWYMUZckblxiIo8EkHRKgUo7M2nvGMlkfsW40y9xZ9zqh7i3A7MQ8CRNTs65t/fkeKHgGhzn28ptbG5t7+R37b39g8OjwnGxpYNIUdakgQhUxyOaCe6zJnAQrBMqRqQnWNubXqd++5EpzQP/DmYh60sy9vmIUwJGGhSKPUlgomSsk/sYkkH8kAwKJafqzIHXiZuREsrQGBR+esOARpL5QAXRuus6IfRjooBTwRK7F2kWEjolY9Y11CeS6X48z57gslGGeBQo83zAc/XvRkyk1jPpmck0qV71UvE/rxvB6Kofcz+MgPl0cWgUCQwBTovAQ64YBTEzhFDFTVZMJ0QRCqYu2y5jGmkIJM7+Wz7sycQ2PbmrrayT1nnVvajWbmuleiVrLI9O0RmqIBddojq6QQ3URBQ9oWf0it6sF+vd+rA+F6M5K9s5QUuwvn4Bk9KilA==</latexit>

Sample for (T-t) Steps

<latexit sha1_base64="Agh1B+L2UzGW7/C+eWOm/pP2w2w=">AAACHnicbVDLSsNAFJ3UV42vWJduBkuhq5KIqMuCG5cV+oI2hsl00o6dScLMRCwhvyLu9EvciVv9EPdO2ixs64GBwzn3zj0cP2ZUKtv+Nkobm1vbO+Vdc2//4PDIOq50ZZQITDo4YpHo+0gSRkPSUVQx0o8FQdxnpOdPb3K/90iEpFHYVrOYuByNQxpQjJSWPKsy5EhNBE9ldp+2My99yDyrajfsOeA6cQpSBQVanvUzHEU44SRUmCEpB44dKzdFQlHMSGYOE0lihKdoTAaahogT6abz7BmsaWUEg0joFyo4V/9upIhLOeO+nsyTylUvF//zBokKrt2UhnGiSIgXh4KEQRXBvAg4ooJgxWaaICyozgrxBAmEla7LNGsQJ1JFHBb/LR/2eWbqnpzVVtZJ97zhXDYu7i6qzXrRWBmcgjNQBw64Ak1wC1qgAzB4As/gFbwZL8a78WF8LkZLRrFzApZgfP0CXlKidA==</latexit>

stj

stj

Sample From p
Amino Acid Types

S

Y

G

1

<latexit sha1_base64="IUbPNaD8icSIpsZrYKXUHldpWBI=">AAACHnicbVDLTgIxFO3gC8cX4tJNIyFhRWaMUZckbtyJiTwSQNIpHai0M5P2jpFM5leMO/0Sd8atfoh7C8xCwJM0OTnn3t6T40WCa3Ccbyu3tr6xuZXftnd29/YPCofFpg5jRVmDhiJUbY9oJnjAGsBBsHakGJGeYC1vfDX1W49MaR4GdzCJWE+SYcB9TgkYqV8odiWBkecnN+l9Amk/eUj7hZJTdWbAq8TNSAllqPcLP91BSGPJAqCCaN1xnQh6CVHAqWCp3Y01iwgdkyHrGBoQyXQvmWVPcdkoA+yHyrwA8Ez9u5EQqfVEemZymlQve1PxP68Tg3/ZS3gQxcACOj/kxwJDiKdF4AFXjIKYGEKo4iYrpiOiCAVTl22XMY01hBJn/y0e9mRqm57c5VZWSfO06p5Xz27PSrVK1lgeHaMTVEEuukA1dI3qqIEoekLP6BW9WS/Wu/Vhfc5Hc1a2c4QWYH39AjAEolk=</latexit>

xtj

Otj

Cα Positions

, Otj

1

CDR

Rt , C

Amino Acid Types

stj
<latexit sha1_base64="tN6b4vl5v2kb1Yz3rtEW5Bwbd5M=">AAACIHicbVDLSgMxFM3UVx1f9bFzEyyFbiwzUtRlwY3LCvYBbS2ZNG1jk5khuSPUYf5F3OmXuBOX+h/uzbSzsNUDgcM59+YejhcKrsFxPq3cyura+kZ+097a3tndK+wfNHUQKcoaNBCBantEM8F91gAOgrVDxYj0BGt5k6vUbz0wpXng38I0ZD1JRj4fckrASP3CUVcSGCsZ6+QuhlM36cf3Sb9QdCrODPgvcTNSRBnq/cJ3dxDQSDIfqCBad1wnhF5MFHAqWGJ3I81CQidkxDqG+kQy3Ytn6RNcMsoADwNlng94pv7eiInUeio9M5lm1cteKv7ndSIYXvZi7ocRMJ/ODw0jgSHAaRV4wBWjIKaGEKq4yYrpmChCwRRm2yVMIw2BxNl/i4c9mdimJ3e5lb+keVZxzyvVm2qxVs4ay6NjdILKyEUXqIauUR01EEWP6Am9oFfr2Xqz3q2P+WjOynYO0QKsrx+HAqMG</latexit>

N

Y

A

1

<latexit sha1_base64="3npdHNHSdIzd1JuH5BTAcBytV9w=">AAACHnicbVDLSsNAFJ3UV42vWJduBkuhq5JIUZcFNy4r2Ae0MUymk3bsTBJmJmIJ+RVxp1/iTtzqh7h30mZhWw8MHM65d+7h+DGjUtn2t1Ha2Nza3invmnv7B4dH1nGlK6NEYNLBEYtE30eSMBqSjqKKkX4sCOI+Iz1/ep37vUciJI3COzWLicvROKQBxUhpybMqQ47URPBUZvepnXnpQ+ZZVbthzwHXiVOQKijQ9qyf4SjCCSehwgxJOXDsWLkpEopiRjJzmEgSIzxFYzLQNEScSDedZ89gTSsjGERCv1DBufp3I0Vcyhn39WSeVK56ufifN0hUcOWmNIwTRUK8OBQkDKoI5kXAERUEKzbTBGFBdVaIJ0ggrHRdplmDOJEq4rD4b/mwzzNT9+SstrJOuucN56LRvG1WW/WisTI4BWegDhxwCVrgBrRBB2DwBJ7BK3gzXox348P4XIyWjGLnBCzB+PoFIiKiUA==</latexit>

N
Cα C

N Cα

N

Orientations

1

Neural Network
Parameterization
C

<latexit sha1_base64="HPqWz7z1XRfeA+JAkn/yZ04NtmM=">AAACHnicbVDLTgIxFO3gC8cX4tJNIyFhRWaMUZckblxiIo8EkHRKByrtzKS9YyCT+RXjTr/EnXGrH+LeArMQ8CRNTs65t/fkeJHgGhzn28ptbG5t7+R37b39g8OjwnGxqcNYUdagoQhV2yOaCR6wBnAQrB0pRqQnWMsb38z81hNTmofBPUwj1pNkGHCfUwJG6heKXUlg5PnJJH1IIO0nj2m/UHKqzhx4nbgZKaEM9X7hpzsIaSxZAFQQrTuuE0EvIQo4FSy1u7FmEaFjMmQdQwMime4l8+wpLhtlgP1QmRcAnqt/NxIitZ5Kz0zOkupVbyb+53Vi8K97CQ+iGFhAF4f8WGAI8awIPOCKURBTQwhV3GTFdEQUoWDqsu0yprGGUOLsv+XDnkxt05O72so6aZ5X3cvqxd1FqVbJGsujU3SGKshFV6iGblEdNRBFE/SMXtGb9WK9Wx/W52I0Z2U7J2gJ1tcvdTSigg==</latexit>

, xtj

Cα

Cα Positions

C
N

C
Cα

Cα
C

N

Orientations

xtj
<latexit sha1_base64="TomYmZABUGmRLXRRSNuo3XQB89s=">AAACIHicbVDLSsNAFJ34rPFVHzs3g6XQjSWRoi4LblxWsFVoa5hMJ+3oTBJmbsQa8i/iTr/EnbjU/3DvpGahrQcGDufcO/dw/FhwDY7zYc3NLywuLZdW7NW19Y3N8tZ2R0eJoqxNIxGpK59oJnjI2sBBsKtYMSJ9wS7929Pcv7xjSvMovIBxzPqSDEMecErASF55tycJjPwgvc+uUzhwMy+9ybxyxak7E+BZ4hakggq0vPJXbxDRRLIQqCBad10nhn5KFHAqWGb3Es1iQm/JkHUNDYlkup9O0me4apQBDiJlXgh4ov7eSInUeix9M5ln1dNeLv7ndRMITvopD+MEWEh/DgWJwBDhvAo84IpREGNDCFXcZMV0RBShYAqz7SqmiYZI4uK/v4d9mdmmJ3e6lVnSOay7R/XGeaPSrBWNldAe2kc15KJj1ERnqIXaiKIH9Iie0Yv1ZL1ab9b7z+icVezsoD+wPr8BaECi9A==</latexit>

Otj
<latexit sha1_base64="+g7/HDFdVD153TIvfhtOH/YwK3o=">AAACIHicbVDLSsNAFJ3UV42v+ti5GSyFbiyJFHVZcOPOClYLbS2T6aQdO5OEmRuhhvyLuNMvcScu9T/cO2mzsOqBgcM59849HC8SXIPjfFiFhcWl5ZXiqr22vrG5VdreudZhrChr0VCEqu0RzQQPWAs4CNaOFCPSE+zGG59l/s09U5qHwRVMItaTZBhwn1MCRuqX9rqSwMjzk4v0NoFDN+0nd2m/VHZqzhT4L3FzUkY5mv3SV3cQ0liyAKggWndcJ4JeQhRwKlhqd2PNIkLHZMg6hgZEMt1LpulTXDHKAPuhMi8APFV/biREaj2RnpnMsurfXib+53Vi8E97CQ+iGFhAZ4f8WGAIcVYFHnDFKIiJIYQqbrJiOiKKUDCF2XYF01hDKHH+3/xhT6a26cn93cpfcn1Uc49r9ct6uVHNGyuifXSAqshFJ6iBzlETtRBFD+gRPaMX68l6td6s99lowcp3dtEcrM9vIr6iyw==</latexit>

1

1

Sample for (t-1) Steps

Prior

<latexit sha1_base64="AxrbkwlsiHPDRp507bp0YnaXElA=">AAACHnicbVDLSsNAFJ3UV42vWJduBkuhq5KIqMuCG5cV7APaGCbTSTt2JgkzE2kJ+RVxp1/iTtzqh7h30mZhWw8MHM65d+7h+DGjUtn2t1Ha2Nza3invmnv7B4dH1nGlI6NEYNLGEYtEz0eSMBqStqKKkV4sCOI+I11/cpP73SciJI3CezWLicvRKKQBxUhpybMqA47U2A/SafaQ2pmXPmaeVbUb9hxwnTgFqYICLc/6GQwjnHASKsyQlH3HjpWbIqEoZiQzB4kkMcITNCJ9TUPEiXTTefYM1rQyhEEk9AsVnKt/N1LEpZxxX0/mSeWql4v/ef1EBdduSsM4USTEi0NBwqCKYF4EHFJBsGIzTRAWVGeFeIwEwkrXZZo1iBOpIg6L/5YP+zwzdU/OaivrpHPecC4bF3cX1Wa9aKwMTsEZqAMHXIEmuAUt0AYYTMEzeAVvxovxbnwYn4vRklHsnIAlGF+/A4SiPg==</latexit>

<latexit sha1_base64="k2mJE0P7P7kuXjQT74eScOo/WnY=">AAACHnicbVDLSsNAFJ3UV42vWJduBkuhq5KIqMuCG3dWsA9oY5hMJ+3YmSTMTMQS8iviTr/EnbjVD3HvpM3Cth4YOJxz79zD8WNGpbLtb6O0tr6xuVXeNnd29/YPrMNKR0aJwKSNIxaJno8kYTQkbUUVI71YEMR9Rrr+5Cr3u49ESBqFd2oaE5ejUUgDipHSkmdVBhypsR+kN9l9amde+pB5VtVu2DPAVeIUpAoKtDzrZzCMcMJJqDBDUvYdO1ZuioSimJHMHCSSxAhP0Ij0NQ0RJ9JNZ9kzWNPKEAaR0C9UcKb+3UgRl3LKfT2ZJ5XLXi7+5/UTFVy6KQ3jRJEQzw8FCYMqgnkRcEgFwYpNNUFYUJ0V4jESCCtdl2nWIE6kijgs/ls87PPM1D05y62sks5pwzlvnN2eVZv1orEyOAYnoA4ccAGa4Bq0QBtg8ASewSt4M16Md+PD+JyPloxi5wgswPj6Bb5FohU=</latexit>

s0j

x0j

Sidechain
Packing

O0j

Figure 3: Illustration of the generative diffusion process. At each step, the network takes the
current CDR state as input and parameterizes the distribution of the CDR’s sequences, positions,
and orientations for the next step. In the end, full-atom structures are constructed by the side-chain
packing algorithm.

Diffusion for Cα Coordinates As the coordinate of an atom could be an arbitrary value, we scale
and shift the coordinates of the whole structure such that the distribution of atom coordinates roughly
match the standard normal distribution. We define the forward diffusion for the normalized Cα
coordinate xj as follows:
 q


t
t · xt−1 , β t I ,
1
−
β
(5)
q xtj | xt−1
=
N
x
pos
pos
j
j
j
q



0 · x0 , (1 − ᾱ0 )I ,
q xtj | x0j = N xtj ᾱpos
(6)
j
pos

t
where βpos
controls the rate of diffusion and its value increases from 0 to 1 as time step goes from
Qt
t
τ
). Using the reparameterization trick proposed by Ho et al., the
0 to t, and ᾱpos
= τ =1 (1 − βpos
generative diffusion process is defined as:



 t 
p xt−1
Rt , C = N xt−1
µp Rt , C , βpos
I ,
(7)
j
j


t

βpos
1  t
µp Rt , C = q
xj − q
G(Rt , C)[j] .
(8)
t
t
αpos
1 − ᾱpos

Here,
q G(·)[j] is a neural network that predicts the standard Gaussian noise ϵj ∼ N (0, I) added
0 x0 (scaled coordinate of amino acid j) based on the reparameterization of Eq.6: xt =
to ᾱpos
j
j
q
q
0 x0 +
0 ϵ . The objective function of training the generative process is the expected
ᾱpos
1 − ᾱpos
j
j

MSE between G and ϵj , which is simplified from aligning distribution p to the posterior q(xt−1
|
j
xtj , x0j ) [Ho et al., 2020]:
 X

1
2
t
t
Lpos = E
ϵj − G(R , C)
.
(9)
j
m
SO(3) Denoising for Amino Acid Orientations We empirically formulate an iterative perturbdenoise scheme for learning and generating amino acid orientations represented by SO(3) elements
[Leach et al., 2022]. Note that we do not use the term diffusion because the formulation does not
strictly follow the framework of diffusion probabilistic models though the overall principle is the
same. Similar to the typical diffusion process, the distribution of orientations perturbed for t steps is
defined as, according to Leach et al. [2022]:

q



t , O0 , 1 − ᾱt
q Otj | O0j = IG SO(3) Otj ScaleRot
ᾱori
(10)
j
ori .
IG SO(3) denotes the isotropic Gaussian distribution on SO(3) parameterized by a mean rotation and
a scalar variance [Leach et al., 2022, Matthies et al., 1970, Nikolayev and Savyolov, 1970]. ScaleRot
modifies the rotation matrix by scaling its rotation angle with the rotation axis fixed [Gallier and Xu,
5

Qt
t
τ
t
2003]. ᾱori
= τ =1 (1 − βori
), where βori
is the variance increases with the step t. The conditional
distribution used for the generation process of orientations is defined as:




t−1
t−1
t
t
t
p Oj R , C = IG SO(3) Oj H(R , C)[j], βori ,
(11)

where H(·)[j] is a neural network that denoises the orientation and outputs the denoised orientation
matrix of amino acid j. Training the conditional distribution requires aligning the predicted orientation
from H(·) to the real orientation. Hence, we formulate the training object that minimizes the expected
discrepancy measured by the inner product between the real and the predicted orientation matrices:

 X
2

1
t
0 ⊺ b t−1
,
(12)
Lori = E
Oj Oj − I
j
m
F
b t−1 = H(·)[j] is the predicted orientation for amino acid j.
where O
j

The Overall Training Objective By summing Eq.4, 9, and 12 and taking the expectation w.r.t. t,
we obtain the final training objective function:


L = Et∼Uniform(1...T ) Lttype + Ltpos + Ltori .
(13)
To train the model, we first sample a time step t and then sample noisy states {stj , xtj , Otj }l+m
j=l+1 ∼ p
by adding noise to the training sample using the diffusion process defined by Eq.2, 6, and 10. We
compute the loss using the noisy data and backpropagate the loss to update model parameters.
3.3

Parameterization with Neural Networks

In this section, we briefly introduce the neural network architectures used in different components of
the diffusion process. The purpose of the networks is to encode the CDR state at a time step t along
t
t
t
with the context structure: {stj , xtj , Otj }l+m
j=l+1 ∪ {si , xi , Oi }i={1...N }\{l+1...l+m} , and then denoises
the CDR amino acid types (F ), positions (G), and orientations (H).
First, we adopt Multiple Layer Perceptrons (MLPs) to generate embeddings for single and pairs
of amino acids. The single amino-acid embedding MLP creates vector ei for amino acid i, which
encodes the information of amino acid types, torsional angles, and 3D coordinates of all the heavy
atoms. The pairwise embedding MLP encodes the Euclidean distances and dihedral angles between
amino acid i and j to feature vectors zij . We adopt IPA [Jumper et al., 2021], an orientation-aware
roto-translation invariant network to transform ei and zij into hidden representations hi , which
aims to represent the amino acid itself and its environment. Next, the representations are fed to
three different MLPs to denoise the amino acid types, 3D positions, and orientations of the CDR,
respectively.
In particular, the MLP for denoising amino acid types outputs a 20-dimensional vector representing
the posterior probabilities. The MLP for denoising Cα coordinates predicts the scaled change of
the coordinate in terms of the current orientation of the amino acid. As the coordinate deviation is
calculated in the local frame, we left-multiply it by the orientation matrix and transform it back to
the global frame. Formally, this can be expressed as ϵ̂j = Otj MLPG (hj ). Predicting coordinate
deviations in the local frame and projecting it to the global frame ensures the equivariance of the
prediction, as when the entire 3D structure rotates by a particular angle, the coordinate deviations also
rotate by the same angle. The MLP for denoising orientations first predicts a so(3) vector [Gallier
and Xu, 2003]. The vector is converted to a rotation matrix Mj ∈ SO(3) right-multiplied to the
b t−1 ← Ot Mj . The
orientation to produce a new mean orientation for the next generative step: O
j
j
proposed networks are equivariant to the rotation and translation of the overall structure:
Proposition 1. For any proper rotation matrix R ∈ SO(3) and any 3D vector r ∈ R3 (rigid
transformation (R, r) ∈ SE(3)), F , G and H satisfy the following equivariance properties:
F (RRt + r, RC + r) = F (Rt , C),
(14)
G(RRt + r, RC + r) = RG(Rt , C),
t

t

H(RR + r, RC + r) = RH(R , C),

(15)
(16)

where RRt +r := {stj , xtj +r, ROtj }l+m
j=l+1 and RC+r := {si , xi + r, ROi }i∈{1...N }\{l+1,...,l+m}
denote the rotated and translated structure. Note that F , G, and H are not single MLPs. Each of
them includes the shared encoder and a specific MLP.
6

3.4

Sampling Algorithms

The sampling algorithm first samples amino acid types from the uniform distribution over 20 classes:
sTj ∼ Uniform(20), Cα positions from the standard normal distribution: xTj ∼ N (0, I3 ), and
orientations from the uniform distribution over SO(3): OTj ∼ Uniform(SO(3)). Note that we
normalize the coordinates of the structure in the same way as training such that Cα positions in the
CDR roughly follow the standard normal distribution. Next, we iteratively sample sequences and
structures from the generative diffusion kernel by denoising amino acid types, Cα coordinates, and
orientations until t = 0. To build a full atom 3D structure, we construct the coordinates of N, Cα , C,
O, and side-chain Cβ (except glycine that does not have Cβ ) according to their ideal local coordinates
relative to the Cα position and orientation of each amino acid [Engh and Huber, 2012]. Based on the
five reconstructed atoms, the rest of the side-chain atoms are constructed using the side-chain packing
function implemented in Rosetta [Alford et al., 2017]. In the end, we adopt the AMBER99 force field
[Lindorff-Larsen et al., 2010] in OpenMM [Eastman et al., 2017] to refine the full atom structure.
In addition to the joint design of sequences and structures, we can constrain partial states for other
design tasks. For example, by fixing the backbone structure (positions and orientations) and sampling
only sequences, we can do fix-backbone sequence design. Another usage is to optimize an existing
antibody. Specifically, we first add noise to the existing antibody for t steps and denoise the perturbed
antibody sequence starting from the t-th step of the generative diffusion process.

4

Experiments

We present the application of our model, named DiffAb4 , in three antibody design tasks: sequencestructure co-design (Section 4.1), antibody sequence design based on antibody backbones (Section
4.2), and antibody optimization (Section 4.3). In Section 4.4, we show how to use our model without
known antibody frameworks bound to the antigen.
4.1

Sequence-Structure Co-design

The dataset for training the model is derived from the SAbDab database[Dunbar et al., 2014]. We first
remove structures whose resolution is worse than 4Å and discard antibodies targeting non-protein
antigens. We cluster antibodies in the database according to CDR-H3 sequences at 50% sequence
identity. We manually select five clusters as the test set, containing 19 antibody-antigen complexes
in total. The test set includes antigens from several well-known pathogens including SARS-CoV-2,
MERS, influenza, and so on. Structures in the remaining clusters are used for training.
To evaluate the performance, we remove the original CDR from the antibody-antigen complex in
the test set and sample both the sequence and structure of the removed region. We set the length
of the CDR to be identical to the length of the original CDR for simplicity. In practice, one can
enumerate different lengths of CDRs. We compare our model to RosettaAntibodyDesign (RAbD)
[Adolf-Bryfogle et al., 2018], an antibody design software based on Rosetta energy functions. For
each model, we draw 100 samples for each CDR. Both the original structures and designed structures
from different methods are refined by OpenMM and Rosetta.
We use the following metrics to evaluate designed antibodies: (1) IMP: is the percentage of designed
CDRs with lower (better) binding energy (∆G) than the original CDR. The binding energy is
calculated by InterfaceAnalyzer in the Rosetta software package [Alford et al., 2017]. (2) RMSD:
is the Cα root-mean-square deviation (RMSD) between the generated structure and the original
structure with only antibody frameworks aligned. (3) AAR: is the amino acid recovery rate measured
by the sequence identity between the reference CDR sequences and the generated sequences [AdolfBryfogle et al., 2018]. Note that different from Jin et al. [2022], we do not use neutralization
prediction models because they are sequence-based and are specified to a limited class of antigens,
which deviates from our goal of developing a general antibody design model.
Table 1 shows that our model (DiffAb) recovers CDR sequences more accurately than RAbD (higher
AAR). The RMSDs of CDRs generated by DiffAb are higher in CDR-H3, which indicates that our
generated samples are more diverse structurally. The IMP score of DiffAb is on par with RAbD in
4

Code and data are available at https://github.com/luost26/diffab.

7

Table 1: Evaluation of the generated antibody CDRs (sequence-structure co-design) by RAbD and
our DiffAb model.
CDR

Method

AAR

RMSD

IMP

CDR

Method

AAR

RMSD

IMP

H1

RAbD
DiffAb

22.85%
65.75%

2.261Å
1.188Å

43.88%
53.63%

L1

RAbD
DiffAb

34.27%
56.67%

1.204Å
1.388Å

46.81%
45.58%

H2

RAbD
DiffAb

25.50%
49.31%

1.641Å
1.076Å

53.50%
29.84%

L2

RAbD
DiffAb

26.30%
59.32%

1.767Å
1.373Å

56.94%
49.95%

H3

RAbD
DiffAb

22.14%
26.78%

2.900Å
3.597Å

23.25%
23.63%

L3

RAbD
DiffAb

20.73%
46.47%

1.624Å
1.627Å

55.63%
47.32%

3

ΔG

R

Reference

Sample 1

Sample 2

Sample 3

ΔG = -10.63
TRGRGLYDYVWGSKDY

ΔG = -15.45
RMSD = 4.26Å
DGSYGSAWYSYVYFDY

ΔG = -12.71
RMSD = 5.36Å
AITYWADDRYYWYFDV

ΔG = -8.99
RMSD = 4.56Å
LIYPEYGNTWYYPMDY

2
1

1

3

2

RMSD (Å)

Figure 4: Examples of CDR-H3 designed by the sequence-structure co-design method and the
distribution of their interaction energy and RMSD. The antigen-antibody template is derived from
PDB:7chf, where the antigen is SARS-CoV-2 RBD. Sample 1 has better complementarity to the
antigen while Sample 3 fits the antigen worse. This could explain their difference in the binding
energy (∆G).
CDR-H3, and lower in other CDRs. However, it should be noted that RAbD optimizes the Rosetta
energy function, which is also used for evaluation. Our model achieves reasonably good binding
energy without explicit supervision signal from Rosetta energy functions. Figure 4 presents three
generated examples of CDR-H3 targeting SARS-CoV-2 RBD. Sample 1 has the lowest binding
energy and it can be observed that it has better complementarity to the antigen. The binding energy
of Sample 3 is higher than the original one and visually, the shape of the CDR does not fit the antigen
well.
4.2

Fix-Backbone Sequence Design and Structure Prediction

In this setting, the backbone structure of CDRs is given and we only need to design the CDR sequence,
which transforms the task into a constrained sampling problem. Fix-backbone design is a common
setting in the area of protein design [Ingraham et al., 2019, Hsu et al., 2022, Anishchenko et al.,
2021, Strokach et al., 2020, Tischer et al., 2020]. For this task, we consider FixBB, a Rosetta-based
sequence design software given CDR backbone structure, as the baseline. We use the AAR metric
introduced in Section 4.1 to evaluate the designed CDRs.
As shown in Table 2, our model achieves better AAR in all the CDRs. This shows that our model
is also powerful in modeling the conditional probability of sequences given backbone structures.
Admittedly, the training data is clustered only by CDR-H3 sequences, so the model might have seen
other CDRs in the test set during training, leading to even higher AAR. However, we believe this is
not an issue as CDRs other than H3 are generally conserved and contribute less to the specificity Xu
and Davis [2000].
Our model can predict CDR structures by fixing the sequence. Table 3 shows that it accurately
predicts the structure of CDR H1, H2, L1, L2, and L3 (RMSD ≤ 1.5Å). The accuracy of CDR-H3
prediction is lower due to the high variability. Figure 5a separately shows the accuracy of different
CDR-H3 lengths. The prediction is generally more accurate for shorter ones. When the CDR-H3
contains more than 10 amino acids, the prediction accuracy drops.

8

Table 2: Comparison of FixBB and DiffAb in terms of amino acid
recovery (AAR) in the fix-backbone CDR design task. DiffAb
achieves higher AAR. The AAR of DiffAb on CDR-H3 is lower
than other CDRs since H3 is much more versatile.

Table 3: The accuracy of CDR
structures predicted by DiffAb
in RMSD.

CDR

Method

AAR

CDR

Method

AAR

CDR

RMSD

H1

FixBB
DiffAb

37.14%
87.83%

L1

FixBB
DiffAb

33.80%
86.63%

H2

FixBB
DiffAb

43.08%
79.70%

L2

FixBB
DiffAb

28.54%
88.91%

H3

FixBB
DiffAb

30.74%
59.48%

L3

FixBB
DiffAb

17.92%
78.69%

H1
H2
H3
L1
L2
L3

0.901Å
1.044Å
3.246Å
1.365Å
1.321Å
1.492Å

RMSD(Å)

t steps
Initial CDR
Optimized CDR

t steps
Backward Diffusion

CDR-H3 Length

(c)
IMP(%)

Forward Diffusion

RMSD(Å)

(b)

SeqID(%)

(a)

Number of Steps

Figure 5: (a) RMSD of predicted CDR-H3 structures grouped by lengths. (b) The antibody optimization algorithm first perturbs the initial CDR for t steps using the forward diffusion process and then
denoises it by the backward diffusion process into the optimized CDR. (c) IMP, RMSD, and SeqID
of the CDRs optimized with different numbers of steps. Dashed lines represent the results of de novo
design. When t = 4, the optimized CDRs reach an IMP score close to de novo CDRs but remain
structurally similar to the original one.

4.3

Antibody Optimization

We use our model to optimize existing antibodies Table 4: Evaluation of optimized CDR-H3s
which is another common pharmaceutical applica- with different numbers of optimization steps.
tion. To optimize an antibody, we first perturb the In contrast to redesigning the CDR, the optiCDR sequence and structure for t steps using the mization method can improve binding energy
forward diffusion process. Then, we denoise the se- while keeping the optimized CDR similar to
quences starting from the (T − t)-th step (t steps the original one. Figure 5c shows the line
remaining) of the generative diffusion process and plot of the results.
obtain a set of optimized antibodies. This process
is illustrated in Figure 5b. We optimize CDR-H3
t
IMP
RMSD
SeqID
of the antibodies in the test set with various t val1 18.52% 1.194Å 92.42%
ues. For each antibody and t, we perturb the CDR
independently 100 times and collect 100 optimized
2 16.67% 1.252Å 91.61%
CDRs different from the original CDR. We report the
4 23.29% 1.290Å 91.16%
percentage of optimized antibodies with improved
8 22.01% 1.447Å 88.78%
binding energy (IMP), RMSD, and sequence identity
16 18.02% 1.759Å 78.43%
(SeqID) of the optimized CDR in comparison to the
32 16.43% 2.623Å 40.58%
original antibody. We also compare the optimized
64 15.47% 3.380Å 27.30%
antibodies with the de novo (t = T = 100) designed
antibodies introduced in Section 4.1. As shown in
T 23.63% 3.597Å 26.78%
Table 4 and Figure 5c, the optimization method could
produce antibodies with improved binding energy measured by the Rosetta energy function. In
contrast to redesigning CDRs, optimization improves binding energy while keeping the optimized
CDR similar to the original one, which is desired in many practical applications.
9

ΔG

Sample 2

Sample 4

ΔG = -64.63
MPGGGAFDY

ΔG = -60.10
RVSGDGFDY

Sample 1

Sample 3

Sample 5

ΔG = -65.54
KIPGSAIDY

ΔG = -64.57
ARVGGGFDY

ΔG = -55.75
SRYGGAFDY

-60 -40 -20

SARS-CoV-2 Omicron RBD
(PDB: 7WVN)

0

20 40

Human Antibody Template
(PDB: 3QHF)

Docking

Antibody-Antigen Framework

Figure 6: A human antibody framework docked to SARS-CoV-2 Omicron RBD using HDOCK.
CDR-H3s are designed based on the docking structure.
4.4

Design Without Bound Antibody Frameworks

In the last experiment, we consider designing antibodies without a known binding pose against the
antigen, a more general and challenging setting. We show that this challenging task could be achieved
with docking software. Specifically, we create an antibody template from an existing antibody
structure by removing its CDR-H3. This is because CDR-H3 is the most variable one and accounts
for most of the specificity, while other CDRs are much more conserved [Xu and Davis, 2000]. Next,
we use HDOCK [Yan et al., 2017] to dock the antibody template to the target antigen to produce the
antibody-antigen complex. In this way, the problem reduces to the original problem so we can adapt
our model to design the CDR-H3 sequence and structure and re-design other CDRs. We demonstrate
using this method to design antibodies for the SARS-CoV-2 Omicron RDB structure (PDB: 7wvn,
residue A322-A590, the structure is not bound to any antibodies). The antibody template is derived
from a human antibody against influenza (PDB: 3qhf). Figure 6 shows the docking structure, five
designed CDR-H3s, and the binding energy distribution. It is hard to confidently conclude that the
generated antibodies are effective without a reference antibody. However, according to the binding
energy distribution, we can still say the generated antibodies are at least reasonable.

5

Conclusions and Limitations

In this work, we propose a diffusion-based generative model for antibody design. Our model is
capable of a wide range of antibody design tasks and can achieve competitive performance. One
main limitation of this work is that it relies on an antibody framework bound to the target antigen.
Therefore, we leave it for future work to design an effective model for generating antibodies without
bound structures. Another limitation is that it remains unclear whether the generated antibodies can
be produced in the wet lab and actually binds to the target. More efforts are needed to design a
biologically effective antibody.

Acknowledgments and Disclosure of Funding
Supported by National Key R&D Program of China No. 2021YFF1201600.

References
Jared Adolf-Bryfogle, Oleks Kalyuzhniy, Michael Kubitz, Brian D Weitzner, Xiaozhen Hu, Yumiko
Adachi, William R Schief, and Roland L Dunbrack Jr. Rosettaantibodydesign (rabd): A general
framework for computational antibody design. PLoS computational biology, 14(4):e1006112,
2018.
Rahmad Akbar, Habib Bashour, Puneet Rawat, Philippe A Robert, Eva Smorodina, Tudor-Stefan
Cotet, Karine Flem-Karlsen, Robert Frank, Brij Bhushan Mehta, Mai Ha Vu, et al. Progress and
10

challenges for the machine learning-based design of fit-for-purpose monoclonal antibodies. In
Mabs, volume 14, page 2008790. Taylor & Francis, 2022a.
Rahmad Akbar, Philippe A Robert, Cédric R Weber, Michael Widrich, Robert Frank, Milena Pavlović,
Lonneke Scheffer, Maria Chernigovskaya, Igor Snapkov, Andrei Slabodkin, et al. In silico proof of
principle of machine learning-based antibody design at unconstrained scale. In Mabs, volume 14,
page 2031482. Taylor & Francis, 2022b.
Rebecca F Alford, Andrew Leaver-Fay, Jeliazko R Jeliazkov, Matthew J O’Meara, Frank P DiMaio,
Hahnbeom Park, Maxim V Shapovalov, P Douglas Renfrew, Vikram K Mulligan, Kalli Kappel,
et al. The rosetta all-atom energy function for macromolecular modeling and design. Journal of
chemical theory and computation, 13(6):3031–3048, 2017.
Ethan C Alley, Grigory Khimulya, Surojit Biswas, Mohammed AlQuraishi, and George M Church.
Unified rational protein engineering with sequence-based deep representation learning. Nature
methods, 16(12):1315–1322, 2019.
Namrata Anand and Tudor Achim. Protein structure and sequence generation with equivariant
denoising diffusion probabilistic models. arXiv preprint arXiv:2205.15019, 2022.
Ivan Anishchenko, Samuel J Pellock, Tamuka M Chidyausiku, Theresa A Ramelot, Sergey Ovchinnikov, Jingzhou Hao, Khushboo Bafna, Christoffer Norn, Alex Kang, Asim K Bera, et al. De novo
protein design by deep network hallucination. Nature, 600(7889):547–552, 2021.
Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured
denoising diffusion models in discrete state-spaces. Advances in Neural Information Processing
Systems, 34:17981–17993, 2021.
Minkyung Baek, Frank DiMaio, Ivan Anishchenko, Justas Dauparas, Sergey Ovchinnikov, Gyu Rie
Lee, Jue Wang, Qian Cong, Lisa N Kinch, R Dustin Schaeffer, et al. Accurate prediction of protein
structures and interactions using a three-track neural network. Science, 373(6557):871–876, 2021.
Kyle A Barlow, Shane O Conchuir, Samuel Thompson, Pooja Suresh, James E Lucas, Markus
Heinonen, and Tanja Kortemme. Flex ddg: Rosetta ensemble-based estimation of changes in
protein–protein binding affinity upon mutation. The Journal of Physical Chemistry B, 122(21):
5389–5399, 2018.
Sidhartha Chaudhury, Sergey Lyskov, and Jeffrey J Gray. Pyrosetta: a script-based interface for
implementing molecular modeling algorithms using rosetta. Bioinformatics, 26(5):689–691, 2010.
Valentin De Bortoli, Emile Mathieu, Michael Hutchinson, James Thornton, Yee Whye Teh, and
Arnaud Doucet. Riemannian score-based generative modeling. arXiv preprint arXiv:2202.02763,
2022.
James Dunbar, Konrad Krawczyk, Jinwoo Leem, Terry Baker, Angelika Fuchs, Guy Georges, Jiye
Shi, and Charlotte M Deane. Sabdab: the structural antibody database. Nucleic acids research, 42
(D1):D1140–D1146, 2014.
Peter Eastman, Jason Swails, John D Chodera, Robert T McGibbon, Yutong Zhao, Kyle A Beauchamp,
Lee-Ping Wang, Andrew C Simmonett, Matthew P Harrigan, Chaya D Stern, et al. Openmm 7:
Rapid development of high performance algorithms for molecular dynamics. PLoS computational
biology, 13(7):e1005659, 2017.
RA Engh and R Huber. Structure quality and target parameters. 2012.
Jean Gallier and Dianna Xu. Computing exponentials of skew-symmetric matrices and logarithms of
orthogonal matrices. International Journal of Robotics and Automation, 18(1):10–20, 2003.
Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair,
Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information
processing systems, 27, 2014.
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in
Neural Information Processing Systems, 33:6840–6851, 2020.
11

Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows
and multinomial diffusion: Learning categorical distributions. Advances in Neural Information
Processing Systems, 34, 2021.
Emiel Hoogeboom, Victor Garcia Satorras, Clément Vignac, and Max Welling. Equivariant diffusion
for molecule generation in 3d. arXiv preprint arXiv:2203.17003, 2022.
Chloe Hsu, Robert Verkuil, Jason Liu, Zeming Lin, Brian Hie, Tom Sercu, Adam Lerer, and Alexander
Rives. Learning inverse folding from millions of predicted structures. bioRxiv, 2022.
John Ingraham, Vikas Garg, Regina Barzilay, and Tommi Jaakkola. Generative models for graphbased protein design. Advances in Neural Information Processing Systems, 32, 2019.
Charles A Janeway, Paul Travers, Mark Walport, and Donald J Capra. Immunobiology. Taylor &
Francis Group UK: Garland Science, 2001.
Wengong Jin, Jeremy Wohlwend, Regina Barzilay, and Tommi S. Jaakkola. Iterative refinement
graph neural network for antibody sequence-structure co-design. In International Conference on
Learning Representations, 2022.
Bowen Jing, Gabriele Corso, Jeffrey Chang, Regina Barzilay, and Tommi Jaakkola. Torsional
diffusion for molecular conformer generation. arXiv preprint arXiv:2206.01729, 2022.
John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger,
Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, et al. Highly accurate
protein structure prediction with alphafold. Nature, 596(7873):583–589, 2021.
Diederik P Kingma and Max Welling.
arXiv:1312.6114, 2013.

Auto-encoding variational bayes.

arXiv preprint

Gideon D Lapidoth, Dror Baran, Gabriele M Pszolla, Christoffer Norn, Assaf Alon, Michael D Tyka,
and Sarel J Fleishman. Abdesign: A n algorithm for combinatorial backbone design guided by
natural conformations and sequences. Proteins: Structure, Function, and Bioinformatics, 83(8):
1385–1406, 2015.
Adam Leach, Sebastian M Schmon, Matteo T Degiacomi, and Chris G Willcocks. Denoising diffusion
probabilistic models on so (3) for rotational alignment. In ICLR 2022 Workshop on Geometrical
and Topological Representation Learning, 2022.
Anders Liljas, Lars Liljas, Goran Lindblom, Poul Nissen, Morten Kjeldgaard, and Miriam-rose Ash.
Textbook of structural biology, volume 8. World Scientific, 2016.
Kresten Lindorff-Larsen, Stefano Piana, Kim Palmo, Paul Maragakis, John L Klepeis, Ron O Dror,
and David E Shaw. Improved side-chain torsion potentials for the amber ff99sb protein force field.
Proteins: Structure, Function, and Bioinformatics, 78(8):1950–1958, 2010.
S Matthies, J Muller, and GW Vinel. On the normal distribution in the orientation space. Textures
and Microstructures, 10, 1970.
Dmitry I Nikolayev and Tatjana I Savyolov. Normal distribution on the rotation group so (3). Textures
and Microstructures, 29, 1970.
RJ Pantazes and Costas D Maranas. Optcdr: a general computational method for the design of
antibody complementarity determining regions for targeted epitope binding. Protein Engineering,
Design & Selection, 23(11):849–858, 2010.
Leonard G Presta. Antibody engineering. Current Opinion in Structural Biology, 2(4):593–596,
1992.
Jeffrey A Ruffolo, Jeffrey J Gray, and Jeremias Sulam. Deciphering antibody affinity maturation with
language models and weakly supervised learning. arXiv, 2021.
Jeffrey A Ruffolo, Lee-Shin Chu, Sai Pooja Mahajan, and Jeffrey J Gray. Fast, accurate antibody
structure prediction from deep learning on massive set of natural antibodies. bioRxiv, 2022a.
12

Jeffrey A Ruffolo, Jeremias Sulam, and Jeffrey J Gray. Antibody structure prediction using interpretable deep learning. Patterns, 3(2):100406, 2022b.
Koichiro Saka, Taro Kakuzaki, Shoichi Metsugi, Daiki Kashiwagi, Kenji Yoshida, Manabu Wada,
Hiroyuki Tsunoda, and Reiji Teramoto. Antibody design using lstm based deep generative model
from phage display library for affinity maturation. Scientific reports, 11(1):1–13, 2021.
Maxim V Shapovalov and Roland L Dunbrack Jr. A smoothed backbone-dependent rotamer library
for proteins derived from adaptive kernel density estimates and regressions. Structure, 19(6):
844–858, 2011.
Chence Shi, Shitong Luo, Minkai Xu, and Jian Tang. Learning gradient fields for molecular
conformation generation. In International Conference on Machine Learning, pages 9558–9568.
PMLR, 2021.
Jung-Eun Shin, Adam J Riesselman, Aaron W Kollasch, Conor McMahon, Elana Simon, Chris
Sander, Aashish Manglik, Andrew C Kruse, and Debora S Marks. Protein design and variant
prediction using autoregressive generative models. Nature communications, 12(1):1–11, 2021.
Ken Shoemake. Uniform random rotations. In Graphics Gems III (IBM Version), pages 124–132.
Elsevier, 1992.
Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised
learning using nonequilibrium thermodynamics. In International Conference on Machine Learning,
pages 2256–2265. PMLR, 2015.
Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution.
Advances in Neural Information Processing Systems, 32, 2019.
Alexey Strokach, David Becerra, Carles Corbi-Verge, Albert Perez-Riba, and Philip M Kim. Fast and
flexible protein design using deep graph neural networks. Cell systems, 11(4):402–411, 2020.
Doug Tischer, Sidney Lisanza, Jue Wang, Runze Dong, Ivan Anishchenko, Lukas F Milles, Sergey
Ovchinnikov, and David Baker. Design of proteins presenting discontinuous functional sites using
deep learning. Biorxiv, 2020.
Brian L Trippe, Jason Yim, Doug Tischer, Tamara Broderick, David Baker, Regina Barzilay, and
Tommi Jaakkola. Diffusion probabilistic modeling of protein backbones in 3d for the motifscaffolding problem. arXiv preprint arXiv:2206.04119, 2022.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing
systems, 30, 2017.
Shira Warszawski, Aliza Borenstein Katz, Rosalie Lipsh, Lev Khmelnitsky, Gili Ben Nissan, Gabriel
Javitt, Orly Dym, Tamar Unger, Orli Knop, Shira Albeck, et al. Optimizing antibody affinity and
stability by the automated design of the variable light-heavy chain interfaces. PLoS computational
biology, 15(8):e1007207, 2019.
Kevin E Wu, Kevin K Yang, Rianne van den Berg, James Y Zou, Alex X Lu, and Ava P Amini.
Protein structure generation via folding diffusion. arXiv preprint arXiv:2209.15611, 2022.
John L Xu and Mark M Davis. Diversity in the cdr3 region of vh is sufficient for most antibody
specificities. Immunity, 13(1):37–45, 2000.
Minkai Xu, Lantao Yu, Yang Song, Chence Shi, Stefano Ermon, and Jian Tang. Geodiff: A geometric
diffusion model for molecular conformation generation. arXiv preprint arXiv:2203.02923, 2022.
Yumeng Yan, Di Zhang, Pei Zhou, Botong Li, and Sheng-You Huang. Hdock: a web server for
protein–protein and protein–dna/rna docking based on a hybrid strategy. Nucleic acids research,
45(W1):W365–W373, 2017.
Jianyi Yang, Ivan Anishchenko, Hahnbeom Park, Zhenling Peng, Sergey Ovchinnikov, and David
Baker. Improved protein structure prediction using predicted interresidue orientations. Proceedings
of the National Academy of Sciences, 117(3):1496–1503, 2020.
13

Ling Yang, Zhilong Zhang, and Shenda Hong. Diffusion models: A comprehensive survey of methods
and applications. arXiv preprint arXiv:2209.00796, 2022.

14


