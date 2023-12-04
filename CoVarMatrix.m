%% CoVar matrix
% To be able to estimate a covariance matrix, we need to parametrize it
% [Covariance matrices must be symmetric and positive semi-definite, this
% could be imposed as constrained optimization but that would be painfully
% slow/difficult to compute. So we want to reparametrize so that we can run
% unconstrained optimization on a vector, and that vector will always
% correspond one-to-one to a covariance matrix.]
% A good intro to this concept is
% Pinheiro & Bates (1996) - Unconstrained parametrization for variance-covariance matrices.
% https://doi.org/10.1007/BF00140873

% Here we work with the method of
% Archakov & Hansen (2021) - A New Parametrization of Correlation Matrices
% https://doi.org/10.3982/ECTA16910

% To understand AH2021, we first need to know that the covariance matrix is
% equivalent to a vector of standard deviations together with the
% correlation matrix.
% So if we can parametrize the correlation matrix, then we combine this
% with a vector of standard deviations, and together these parametrize the
% covariance matrix. AH2021 show how to parametrize the correlation matrix
% (the concept is simple, but when reading the AH2021 paper it is very easy 
% to get lost in the details).

% We will proceed in two parts. First we show how we can turn an
% (arbitrary) covariance matrix into a vector that parametrizes it.
% Second, we show how to (re)construct the covariance matrix from the
% vector that parametrizes it. The second part here is what you are
% actually likely to be wanting to do, as it will typically be a step in
% the function/model that you want to solve/evaluate.

%% Let's do an example, we will start off with a Covariance matrix.
% We aim to first turn this into a vector of standard deviations, plus a
% correlation matrix. Then use AH2021 to turn the correlation matrix into a
% vector of parameters. Combining the vector of standard deviations with
% the vector of parameters from AH2021 applied to correlation matrix, we
% get a vector of parameters that parametrize the covariance matrix, we
% denote this vector CoVarParametrization.

% Covariance matrix
CoVarMatrix=[0.7,0.3;0.3,0.6];
% This is totally arbitrary

% Before we start, just double-check that it is symmetric and positive
% semidefinite (that it is a Covariance matrix)
% https://au.mathworks.com/help/matlab/math/determine-whether-matrix-is-positive-definite.html
try chol(CoVarMatrix)
    disp('Matrix is symmetric positive definite.') % good
catch ME
    error('Matrix is not symmetric positive definite') % bad, is not a covariance matrix
end

% We can use Matlab function cov2corr() to get the correlation matrix (and vector of standard deviations)
% from our covariance matrix.
[StdDevVector,CorrMatrix] = cov2corr(CoVarMatrix);
% Note: the diagonals of the correlation matrix are always ones by definition/construction

% The key contribution of AH2021 is how we can parametrize this correlation matrix as a vector.
% AH2021, page 1701 "[we] parametrize correlation matrices using the
% off-diagonal elements of logC [C is correlation matrix]"
logC=logm(CorrMatrix); % logC
temp=tril(logC,-1); % off-diagonal elements (as matrix)
AHcorrvector=temp(temp~=0); % drop the zeros (so now as column vector)
AHcorrvector=AHcorrvector'; % as a row vector

% We now have the vector of standard deviations, and a vector that
% parametrizes the correlation matrix. Combined these give us a vector that
% parametrizes the covariance matrix.
CoVarParametrization=[StdDevVector,AHcorrvector];

%% Now, (re)construct covariance matrix from the vector that parametrizes it
% When estimating the covariance matrix this is what we will want to do. We
% can do an unconstrained optimization on the vector that parametrizes the
% covariance matrix, and then one of the steps inside our routine will be
% to take this vector and construct the covariance matrix so that we can
% solve and evaluate our model.
% So the following are the steps you need to do inside your function that
% you want to optimize.

% First, just declare the size of the covariance matrix that we are trying
% to create
n=size(CoVarMatrix,1); % Obviously you cannot do this actual line in your implementation, but n is going to be known by you so you can just declare it directly, e.g. n=3;

% Trivially first step, split the vector into the part relating to std
% deviation and the part relating to the correlation matrix
StdDevVector2=CoVarParametrization(1:n);
AHcorrvector2=CoVarParametrization(n+1:end);

% We use AH2021 to turn this later vector into the correlation matrix
tol_value=10^(-9); % AH2021 require a tolerance (between 10^(-4) and 10^(-14))
[CorrMatrix2, iter_number ] = GFT_inverse_mapping(AHcorrvector2, tol_value);
% iter_number is the number of iterations it took to converge (to the tolerance declared in tol_value)
% Note: AH2021 provide GFT_inverse_mapping() in their Online Appendix (for multiple
% programming languages). The version used here is a lightly modified/cleaned version of theirs.

% And then use Matlab corr2cov() to convert the vector of std deviations
% and correlation matrix into the covariance matrix
CoVarMatrix2 = corr2cov(StdDevVector2,CorrMatrix2);

% Voila!!!
% Easy enough :)



%% Take a look to convince ourselves everything works 
CoVarMatrix
CoVarMatrix2
max(max(abs(CoVarMatrix-CoVarMatrix2)))
max(max(abs(CorrMatrix-CorrMatrix2)))
% Note, that the std dev vector is same is trivially true, likewise for AHcorrvector


%% A few final comments
% AH2021 emphasize that their method for parametrizing the (correlation and
% hence) covariance matrix has four nice properties:
% (1) any non-singular covariance matrix, Sigma, maps to a unique vector g=gamma(Sigma) \in R^d.
% (2) any vector, g \in R^d, maps to a unique covariance matrix Sigma=gamma^(-1)(g)
% (3) the parametization, g=gamma(Sigma), is "invariant" to the ordering of the variables that define Sigma
% (4) the elements of g are easily interpretable
%
% [g is CoVarParametrization in the above codes, Sigma is CoVarMatrix is the above codes]
% [gamma() is about constructing CoVarParametrization from CoVarMatrix; mostly implemented using cov2corr() and logm()]
% [gamma^(-1)() is about constructing CoVarMatrix from CoVarParametrization; mostly implemented using GFT_inverse_mapping() and corr2cov()]

% Pinheiro & Bates (1996) show a variety of other ways to parametrize the
% covariance matrix. All of them satisfy (1) and (2), but few satisfy (4).
% They also emphasize another important property, if we do some kind of
% likelihood/moment estimation then we will get standard errors/confindence
% intervals for the parametrization parameters. We are however interested
% in standard errors/confidence intervals for things like the standard
% deviation of shocks, and the correlations/covariances (not in the parametrization 
% parameters per se). AH2021 is nice in that as soon as we get the standard 
% errors/confidence intervals for our parametrization parameters, we already 
% have them for standard errors and getting them for correlations should be easy enough.

% AH2021 also discuss that their parametrization appears to avoid skewness
% in the parameters (which would make finding the optimum more computationally difficult).
% When the covariance matrix is 2x2 their approach inherits properties of
% the 'Fisher transformation' (see their paper) and this means it is
% symmetric in the parameters which further helps make finding the optimum
% computationally easier.

%% How to apply this
% Set up your optimization routine to take CoVarParametrization as input to 
% be optimized (part of the argmax).
% Inside your function, covert CoVarParametrization into CoVarMatrix, then
% use CoVarMatrix to solve/evaluate your model (so essentially a copy-paste
% of the second half of the code here).




