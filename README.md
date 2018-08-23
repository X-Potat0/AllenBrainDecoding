# AllenBrainDecoding

These scripts use the SDK from the Allen Brain Observatory to download and analyse their publicly available dataset 
of two-photon imaging experiments performed in primary and secondary visual cortex. The SDK is available for download here:
http://observatory.brain-map.org/visualcoding/sdk/index

Specifically, these scripts use several decoding approaches to decode stimulus features from the neural population activity. 
Using a jackknifing procedure, the contribution of single neurons to the fidelity of the decoder is determined. Subsequently, 
greedy decoding can be performed by decoding using the 'best' neuron and sequentually add neurons according to their decoding
contribution. The hypothesis that will be tested is that the 'best' neurons exhibit noise correlations which are parallel to 
decision boundaries (Montijn & Meijer et al., 2016) whereas the 'worst' neurons show so-called differential correlations 
(Moreno-Bote et al., 2014). 

Montijn, J.S., Meijer, G.T., Lansink, C.S., and Pennartz, C.M.A. (2016). Population-Level Neural Codes Are Robust to 
Single-Neuron Variability from a Multidimensional Coding Perspective. Cell Reports 16, 2486–2498.

Moreno-Bote, R., Beck, J., Kanitscheider, I., Pitkow, X., Latham, P., and Pouget, A. (2014). 
Information-limiting correlations. Nature Neuroscience 17, 1410–1417.

