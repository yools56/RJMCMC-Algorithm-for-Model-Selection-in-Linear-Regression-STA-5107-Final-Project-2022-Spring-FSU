%% Simulate a dataset with the following Matlab code

m=10; % The given number of predictors.
n_0=ceil(rand*m); %% The selected number of predictors. n << m, where n_0 : true value of 'n' in this simulation.
%% The above 'n_0' would be True Value of n.
%% 'n' would be RANDOM (Variable) whose realized value is non-negative integer.
%% 'rand' would provide a randon number from U(0,1) distribution.

k=10; % The number of independent measurements (=observations)
Sigma_0=0.2; 
Sigma_p=0.3;
Mu_b=2*ones(n_0,1); %% 'n_0' x 1 vector
b=Mu_b+Sigma_p*randn(n_0,1);  %% 'n_0' x 1 vector
X=5*randn(k,m); %% 'k' x 'm' matrix.
y=X(:,1:n_0)*b+Sigma_0*randn(k,1); % 10(=k) X 1 vector


%% Select n_first_star, denoted by 'n^{*_1}': the first candidate number from the probability Prior_prob_on_n = 1/m;

n_first_star=ceil(rand*m);


Sigma_r=0.2;

%% Make the row vector that contains all of the n^{*} values simulated from f(n) = 1/m as follow:
%% and the first element in the above row vector would be 'n^{*_1}'=n_first_star
Set_of_n_values(1) = n_first_star;

%% 'for' loop to implement RJMCMC Algorithm (to get 100,000 posterior samples) starting from the below line.

for i=1:100000 
    
%%% Select n_star: candidate number from the probability Prior_prob_on_n = 1/m;

n_star=ceil(rand*m); %% This value would be changed for every 'for' loop.

%%%%% Case 1) n_star >= n_0
  if(n_star >= Set_of_n_values(i))
%%% Let's generate a random vector: u ~ mvnrnd(0,Sigma_r * I_n_star). The
%%% candidate coefficient vector, denoted by b_n_star as follows:

u=mvnrnd(repelem(0,n_star),Sigma_r^2*eye(n_star));

u=reshape(u,[],1);

u1 = u(1:Set_of_n_values(i),:);

u2 = u(Set_of_n_values(i)+1:size(u,1),:);


Mu_b_n=2*ones(Set_of_n_values(i),1); %% 'Set_of_n_values(i)'(=ith element) x 1 vector

b_n=Mu_b_n+Sigma_p*randn(Set_of_n_values(i),1);  %% 'Set_of_n_values(i)'(=ith element) x 1 vector




b_with_Length_increased=[b_n ; repelem(0,(n_star-Set_of_n_values(i)))']; %% [b_n0 ,0...]

b_n_star = b_with_Length_increased + u; %% 'n_star' x 1 vector


%% Compute the likelihoods

Likeli_h1_u = (1/sqrt(2*pi*Sigma_r^2))^n_star * exp((-1/(2*Sigma_r^2))*norm(u)^2);

Likeli_h2_u1 = (1/sqrt(2*pi*Sigma_r^2))^Set_of_n_values(i) * exp((-1/(2*Sigma_r^2))*norm(u1)^2);

%% Define the Mu_b_n_star 

Mu_b_n_star=2*ones(n_star,1); %% 'n_star' x 1 vector

%% Calculate the Likelihood for n_star

Likeli_n_star = (1/sqrt(2*pi*Sigma_0^2))^k * exp((-1/(2*Sigma_0^2))*norm(y-X(:,1:n_star)*b_n_star)^2);

%% Calculate the Prior prob. on b_n_star

Prior_prob_on_b_n_star = (1/sqrt(2*pi*Sigma_p^2))^n_star * exp((-1/(2*Sigma_p^2))*norm(b_n_star-Mu_b_n_star)^2);

%% Prior prob. on n_star

Prior_prob_on_n_star = 1/m;

%% Posterior prob. on n_star

Posterior_prob_on_b_n_star_and_n_star = Likeli_n_star * Prior_prob_on_b_n_star * Prior_prob_on_n_star;


%% Calculate Acceptance-Rejection function value.

% E1 & E2

E1 = (1/(2*Sigma_0^2))*norm(y-X(:,1:n_star)*b_n_star)^2;
E2 = (1/(2*Sigma_0^2))*norm(y-X(:,1:Set_of_n_values(i))*b_n)^2;

MH_Ratio = (exp(-(E1-E2))*(2*pi*Sigma_p^2)^((Set_of_n_values(i)-n_star)/2)*exp((-1/(2*Sigma_p^2))*(norm(b_n_star-Mu_b_n_star)^2-norm(b_n-Mu_b_n)^2))*Likeli_h2_u1)/Likeli_h1_u;   %% MH Ratio.

Rho = min(MH_Ratio,1); 
    
%% Update Set_of_n_values(i+1) as following logic:
    if rand < Rho
        Set_of_n_values(i+1) = n_star;
    else
        Set_of_n_values(i+1) = Set_of_n_values(i);
    end
    
  else     
  %% Case 2) n_star < n_0

%%% Let's generate a random vector: u ~ mvnrnd(0,Sigma_r * I_n_star). The
%%% candidate coefficient vector, denoted by b_n_star as follows:

u1=mvnrnd(repelem(0,n_star),Sigma_r^2*eye(n_star));

u1=reshape(u1,[],1);


Mu_b_n=2*ones(Set_of_n_values(i),1); %% 'Set_of_n_values(i)'(=ith element) x 1 vector

b_n=Mu_b_n+Sigma_p*randn(Set_of_n_values(i),1);  %% 'Set_of_n_values(i)'(=ith element) x 1 vector


b_n_to_the_1=b_n(1:n_star,:); %% b_n^{1}

b_n_to_the_2=b_n(n_star+1:size(b_n,1),:); %% b_n^{2}

b_n_star = b_n_to_the_1 + u1; %% 'n_star' x 1 vector

u=[u1 ; b_n_to_the_2];



%% Compute the likelihoods

Likeli_h2_u = (1/sqrt(2*pi*Sigma_r^2))^Set_of_n_values(i) * exp((-1/(2*Sigma_r^2))*norm(u)^2);

Likeli_h1_u1 = (1/sqrt(2*pi*Sigma_r^2))^n_star * exp((-1/(2*Sigma_r^2))*norm(u1)^2);

%% Define the Mu_b_n_star

Mu_b_n_star=2*ones(n_star,1); %% 'n_star' x 1 vector

%% Calculate the Likelihood for n_star

Likeli_n_star = (1/sqrt(2*pi*Sigma_0^2))^k * exp((-1/(2*Sigma_0^2))*norm(y-X(:,1:n_star)*b_n_star)^2);

%% Calculate the Prior prob. on b_n_star

Prior_prob_on_b_n_star = (1/sqrt(2*pi*Sigma_p^2))^n_star * exp((-1/(2*Sigma_p^2))*norm(b_n_star-Mu_b_n_star)^2);

%% Prior prob. on n_star

Prior_prob_on_n_star = 1/m;

%% Posterior prob. on n_star

Posterior_prob_on_b_n_star_and_n_star = Likeli_n_star * Prior_prob_on_b_n_star * Prior_prob_on_n_star;


%% Calculate Acceptance-Rejection function value.

% E1 & E2

E1 = (1/(2*Sigma_0^2))*norm(y-X(:,1:n_star)*b_n_star)^2;
E2 = (1/(2*Sigma_0^2))*norm(y-X(:,1:Set_of_n_values(i))*b_n)^2;

MH_Ratio = (exp(-(E1-E2))*(2*pi*Sigma_p^2)^((Set_of_n_values(i)-n_star)/2)*exp((-1/(2*Sigma_p^2))*(norm(b_n_star-Mu_b_n_star)^2-norm(b_n-Mu_b_n)^2))*Likeli_h2_u)/Likeli_h1_u1;   %% MH Ratio.

Rho = min(MH_Ratio,1);  

%% Update Set_of_n_values(i+1) as following logic:
  if rand < Rho
        Set_of_n_values(i+1) = n_star;
    else
        Set_of_n_values(i+1) = Set_of_n_values(i);
    end
end
end


%% Draw the Histogram of 'n' values (from 'Set_of_n_values(2:100001)' consisting 100,000 Posterior Samples obtained from above RJMCMC Algorithm.)
histogram(Set_of_n_values(2:100001),'Normalization','probability'); %% We want to get normalized probability.
xlabel('n')
ylabel('Normalized Probability')
%% True value of n=n_0, and initial value of n^{*}=n^{*_1}
title(sprintf("True value of n(=n_0): %d, and initial value of n^{*}: %d", n_0, n_first_star));
%% Proportion of Correct Sampling denotes the proportion of 'n_0' sampled from the posterior density.
subtitle(sprintf("Proportion of Correct Sampling: %.2f%%",((sum(Set_of_n_values(2:100001) == n_0)/length(Set_of_n_values(2:100001)))*100)));
