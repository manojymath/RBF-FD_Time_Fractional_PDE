%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Title of the article: Highly accurate RBF-FD based method for 
%%%%%%%% time-fractional Black-Scholes PDE modeling option pricing problems
%%%%%%%% Authors: A. Sreedhar, Manoj Kumar Yadav, Chirala Satyanarayana 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% To compute the optimal epsilon value for 1D-Diffusion
%%%%%%%%%%%% equation using the LTE expressions for the operator of PDE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% % % % % % To Find optimal shape parameter (ep) 
% % % % % % Minimizes both local truncation error and global error
% % % % % Inputs:
% % N - Number of grid points
% % % % % Outputs: Errors and corresponding 'ep' values and plots
% % Err_1 ---------  ep minimizing by maximun error
% %   LTE ---------  ep minimizing LTE error
% % Err_2 ---------  ep minimizing Global error

%% % % % % % % The function file to find Max, LTE and Global Error
function [Err_1, LTE, Err_2] = Opt_ep_1D_Diff_Eqn(ep,T) % % % with Exact

%% % % % % %  % Intial considerations
% % % % % % % % Problem setup
a = 0; b = 1; % % % Interval of the Problem
N = 12; % % % % Number of divisions along Spatial domain
x = linspace(a, b, N+1)'; % % % % % Spatial points
h = x(2)-x(1); % % % % % Spatial step size

%% % % % % % % % Time-stepping parameters
alpha = 0.1; % % % % % Fractional order
tau = 0.0001;  % % % % % Time step size
% % T = 1.0; % % % % % Final time
M=ceil(T/tau); % % % % % Number of time steps
t = linspace(0, T, M+1)'; % % % % % Time points

%% % % % % % OneD pure diffusion problem 
% % % % % % % Example 1(Given functions)
f=@(x,t) ((gamma(4+alpha)*(t.^3))/6).*(sin(pi*x))...
           +pi^2*t.^(3+alpha).*sin(pi*x);
uexact=@(x,t) (t.^(3+alpha)).*(sin(pi*x));
b_1=@(t) 0; b_last= @(t) 0;

%% % % % % Discretization of Spatial derivative Using GA RBF-FD Method
% % % % % % derived by our self.
D2=zeros(N+1,N+1);
% % % % Second derivative weights
beta1 = 1/(h^2)+(5*ep^2)/6+(ep^4*h^2)/36-(13*ep^6*h^4)/108+(19*ep^8*h^6)/1296;
beta2 = -2/(h^2)-(5*ep^2)/3-(ep^4*h^2)/18+(13*ep^6*h^4)/54-(19*ep^8*h^6)/648;
beta3 = 1/(h^2)+(5*ep^2)/6+(ep^4*h^2)/36-(13*ep^6*h^4)/108+(19*ep^8*h^6)/1296;
% % % % % % % % Assign row-wise % % % % % % % %
for l=2:N
    D2(l, l-1:l+1) = [beta1, beta2, beta3];
end
D2(1,1)=1; D2(1,2:N)=0; D2(N+1,1:N)=0; D2(N+1,N+1)=1;

%% % % % % % % Time Discretization (L1-formula)
% % % % Fractional memeory terms 
a_coeff=zeros(M,1);
for k=1:M
    a_coeff(k)=(k)^(1-alpha)-(k-1)^(1-alpha);
end
% % % % % Fractional constant
lambda1=(tau^(alpha))*(gamma(2-alpha));

% % % % % % Intialization of the solution
u=zeros(N+1,M+1);
% % % % % To find the First Level Solution
u(:,1) = uexact(x,0);
B = eye(N+1)-lambda1*D2;
F=u(:,1)+lambda1*[0;f(x(2:N),t(2));0];
u(:,2) = B\F;
u(1,2)=b_1(t(2)); u(N+1,2)=b_last(t(2));

% % % % % To find the higher Level Solutions from 2 to M
for m=2:M
    s=u(:,m);
    % % % % % % % loop over all values of time levels 3 to M
    for k=m:-1:2
        s=s-a_coeff(k)*(u(:,m+2-k)-u(:,m+1-k));
    end
    G=s+lambda1*[0;f(x(2:N),t(m+1));0];
    G(1)=b_1(t(m+1)); G(N+1)=b_last(t(m+1));
    u(:,m+1)=B\G;
    u(1,m+1)=b_1(t(m+1)); u(N+1,m+1)=b_last(t(m+1));
end

%% % % % % Finding the Error
Err_1=norm(uexact(x,t(M+1))-u(:,M+1),inf);

%% =========== Truncation Errors for All Time Levels =========== %%

% % % % % Preallocate truncation error matrices
T2 = zeros(N+1, M+1); % % % % For 2nd derivative

% % % % Loop over time levels for interior points
for m = 2:M
    U_current = u(:,m);

    % % % % % Compute spatial derivatives
    Uxx   = D2*U_current;
    Uxxxx = D2*Uxx;

    % % % % Truncation Error for second derivative(for-space)
    T2(:,m) = h^2*((5/6)*ep^2*Uxx+(1/12)*Uxxxx);
end

% % % % % % Intialization of the Global error
Globl_err1 = zeros(N+1,M+1);

% % % % % To find for next step
Rhs2 = lambda1*T2(:,2);
% % % % Imposing the Boundary conditions
Rhs2(1)=b_1(t(2)); Rhs2(N+1)=b_last(t(2));
% % % % % % To find the error for the Second Level
Globl_err1(:,2) = B\Rhs2;
% % % % Imposing the Boundary conditions
Globl_err1(1,2)=b_1(t(2)); Globl_err1(N+1,2)=b_last(t(2));

% % % % % %  Compute Global Error from time level 3 to M+1
for m = 2:M
    s1 = Globl_err1(:,m);

    % % % % % %  Apply fractional accumulation similar to 'u' calculation
    for k = m:-1:2
        s1 = s1-a_coeff(k)*(Globl_err1(:,m+2-k)-Globl_err1(:,m+1-k));
    end
    R1 = s1-lambda1*T2(:,m+1);
    R1(1)=b_1(t(m+1)); R1(N+1)=b_last(t(m+1));
    % % % % % %  Solve for next Global Error step
    Globl_err1(:,m+1) = B\R1;
    Globl_err1(1,m+1)=b_1(t(m+1)); Globl_err1(N+1,m+1)=b_last(t(m+1));
end

%% % % To find the Global Error
Err_2 = norm(Globl_err1,inf); % Infinity norm of global error
% Err_2 = norm(Globl_err1(:,M+1),inf); % Infinity norm of global error

%% % % % % % -------------------------------
% % % % % % % Local truncation error function
% % % % % % -------------------------------
U = u(:,M+1);
u_double = D2*U;  u_fourth = D2*D2*U;

% % % % % % Intilization of LTE Errors
LTE1 = zeros(N+1,1);
for i = 2:N
    a1 = (5/6)*u_double(i);  a2 = (1/12)*u_fourth(i);
    LTE1(i) = (h^2)*(a1*ep^2+a2);
end
LTE = norm(LTE1, inf);

%% % % % % % % To end the function file
end
