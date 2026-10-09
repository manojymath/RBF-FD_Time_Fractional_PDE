%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Title of the article: Highly accurate RBF-FD based method for 
%%%%%%%% time-fractional Black-Scholes PDE modeling option pricing problems
%%%%%%%% Authors: A. Sreedhar, Manoj Kumar Yadav, Chirala Satyanarayana 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% To compute the optimal epsilon value for 1D-Diffusion
%%%%%%%%%%%% equation using the LTE expressions for the operator of PDE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all 

%% % % % % % % % Code Procedure Starts...

%% % % Define the epsilon values
ep=linspace(1/100,2.0,100); Np=length(ep);

% % % % Intialization of requried solutions
Max_error=ones(Np,1); LTE = ones(Np,1);  Globl=ones(Np,1);

% % % % % % To find the solution for ep=0;  
T= 1.0; % % % % % Final time 
[Max_error_FD, LTE_FD, Globl_FD] = Opt_ep_1D_Diff_Eqn(0,T);
loglog(ep,Max_error*Max_error_FD,'k--');
hold on;

for k=1:Np
    [Max_error(k), LTE(k), Globl(k)] = Opt_ep_1D_Diff_Eqn(ep(k),T); 
    K1=k  % % % % Debugging step to verify progress
end

% % % % % Estimated Optimal Shape Parameter from RBF-HFD
[Min_Max_error,idx]=min(Max_error);
opt_ep1=ep(idx);

% % % % % % Estimated Optimal Shape Parameter from LTE
[Min_Max_LTE,idx1]=min(LTE);
est_opt_ep1=ep(idx1);

% % % % % Estimated Optimal Shape Parameter from Global_Error
[Min_Max_Globl,idx_Globl]=min(Globl);
opt_ep_Globl=ep(idx_Globl);

Ep_ex1 = opt_ep1;      % % % % % (From RBF-FD optimised epsilon)
Ep_ap1 = est_opt_ep1;  % % % % % (From-LTE optimised epsilon)
Ep_Gl1 = opt_ep_Globl; % % % % % (From-Global optimised epsilon)

Error_ep=abs(est_opt_ep1-opt_ep1); % % %(differnce of LTE and RBF-FD ep)

  [Max_err_Ex1, ~,  ~] = Opt_ep_1D_Diff_Eqn(Ep_ex1,T); % % (For RBF-FD)
  [Max_err_Ap1, ~,  ~] = Opt_ep_1D_Diff_Eqn(Ep_ap1,T); % % (For LTE)
[Max_err_Globl1, ~, ~] = Opt_ep_1D_Diff_Eqn(Ep_Gl1,T); % % (For Global)

%% % % plotting the required curves 
loglog(ep,Max_error,'k-');
hold on;
loglog(ep,LTE,'b-');
hold on;
loglog(ep,Globl, 'r-'); 
hold on;
grid on
legend({'$FD (ME_\infty)$', '$RBF-FD (ME_\infty)$', '$LTECF (TE_\infty)$',...
        '$Global (GE_\infty)$'},'Interpreter', 'latex', 'Location',...
        'NorthWest');
xlabel('$\epsilon$', 'Interpreter', 'latex');
ylabel('$Error$', 'Interpreter', 'latex');
grid on;
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
