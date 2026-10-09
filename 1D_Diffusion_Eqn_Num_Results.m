%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Title of the article: Highly accurate RBF-FD based method for
%%%%%%%% time-fractional Black-Scholes PDE modeling option pricing problems
%%%%%%%% Authors: A. Sreedhar, Manoj Kumar Yadav, Chirala Satyanarayana
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% To compute the variable epsilon value and to find the maximum
%%%%%%%%%%%% error for 1D-Diffusion equation (Example.1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% % % % Code procedure starts from here...
clc
clear all

% % % % % % % % Problem setup
a = 0; b = 1; % % % % Interval of the Problem

% % % % % % % % Time-stepping parameters
alpha = 0.1;  % % % % % Fractional order
tau = 0.0001; % % % % % Time step size
T = 1.0; % % % % Final time
M = ceil(T/tau);  % % % % % Number of time steps
t = linspace(0, T, M+1)'; % % % % % Time points

%% % % % % % % Given functions

% % % % Forcing function
f = @(x,t) ((gamma(4+alpha)*(t.^3))/6).*(sin(pi*x))+...
    pi^2*t.^(3+alpha).*sin(pi*x);
% % % % Exact Solutions
uexact=@(x,t) (t.^(3+alpha)).*(sin(pi*x));

% % % % Boundary conditions
b_1=@(t) 0; b_last= @(t) 0;

% % % % % Array of N values
N_values = [3, 6, 12, 24];

%% % % % Intializations
Err_1FD = zeros(size(N_values)); % % % % Intialization of errors for FD
Err_1 = zeros(size(N_values)); % % % % Intialization of errors for RBF-FD

%% Loop to contnue the problem for solution using FD matrices.
for k = 1:length(N_values)

    % % % % % Spatial Parameters
    N = N_values(k);
    x = linspace(a, b, N+1)'; % % % % Number of points along X axis
    h = x(2)-x(1);  % % % % % Step size along the spatial direction

    % % % % % % Discretization of Spatial derivative Using FD Method
    FD2=zeros(N+1,N+1);

    % % % % Second derivative weights
    betaFD1 = 1/(h^2); betaFD2 = -2/(h^2);  betaFD3 = 1/(h^2);
    for i=2:N
        FD2(i,i-1:i+1)=[betaFD1   betaFD2    betaFD3];
    end
    FD2(1,1)=1; FD2(1,2:N)=0; FD2(N+1,1:N)=0; FD2(N+1,N+1)=1;

    % % % % % % % % % % % Compute time discretization (L1 formula)
    a_coeffs = zeros(M,1);
    for q = 1:M
        a_coeffs(q) = q^(1-alpha)-(q-1)^(1-alpha);
    end
    lambda1 = (tau^(alpha))*(gamma(2-alpha));

    % % % % % % % % % % % Initialize solution
    uFD = zeros(N+1, M+1);

    % % % % % % % % Initial condition
    uFD(:,1) = uexact(x,0);

    % % % % % Define operator Matrix for FD
    B_FD = eye(N+1)-(lambda1)*FD2;

    % % % % % % To find the next level solution (i.e, Second level)
    F_FD=uFD(:,1)+lambda1*[0;f(x(2:N),t(2));0];
    uFD(:,2)=B_FD\F_FD;

    % % % % % % Append the boundary conditions
    uFD(1,2)=b_1(t(2)); uFD(N+1,2)=b_last(t(2));

    % % % % % % Loop over all the time levels
    for m=2:M
        sFD=uFD(:,m);
        for pFD=m:-1:2
            sFD=sFD-a_coeffs(pFD)*(uFD(:,m+2-pFD)-uFD(:,m+1-pFD));
        end
        GFD=sFD+lambda1*[0;f(x(2:N),t(m+1));0];
        GFD(1)=b_1(t(m+1)); GFD(N+1)=b_last(t(m+1));
        uFD(:,m+1)=B_FD\GFD;
        uFD(1,m+1)=b_1(t(m+1)); uFD(N+1,m+1)=b_last(t(m+1));

        % % % % % % Calculating the Error with FD
        Err_1FD(k) = norm(uexact(x,t(M+1))-uFD(:,M+1),inf);

    end

    %% % % % % To find the Higher order derivative to find epsilon value
    % % % % % Recursive matrix multiplication for 2nd and 4th derivatives
    U  = uFD(:, M+1);
    u2 = FD2*U;  % % % 2nd derivative: u''(x)
    u4 = FD2*u2; % % % 4th derivative: u''''(x) = (FD2^2)*U

    % % % % % Variable epsilon: first and last set to zero,
    % % % % % Calculated for interior nodes using the GA based RBF-FD LTE
    ep = zeros(1, N+1);
    ep(1)   = 0;  % % % % First value set to zero
    ep(N+1) = 0;  % % % % Last value set to zero
    for i = 2:N   % % % % Rest of the values calculated
        ep(i) = -(u4(i))/(10*u2(i)); % % % Formula by GA based RBF-FD LTE
    end

    % % % % % % Discretization of Spatial derivative Using GA RBF-FD Method
    % % % % % % derived by our self.
    D2=zeros(N+1,N+1);
    for i = 2:N
        % % % % Second derivative weights computed locally for each ep(i)
        beta1 = 1/(h^2)+(5*ep(i))/6+(ep(i)^2*h^2)/36-(13*ep(i)^3*h^4)/108+(19*ep(i)^4*h^6)/1296;
        beta2 = -2/(h^2)-(5*ep(i))/3-(ep(i)^2*h^2)/18+(13*ep(i)^3*h^4)/54-(19*ep(i)^4*h^6)/648;
        beta3 = 1/(h^2)+(5*ep(i))/6+(ep(i)^2*h^2)/36-(13*ep(i)^3*h^4)/108+(19*ep(i)^4*h^6)/1296;
        D2(i, i-1:i+1) = [beta1, beta2, beta3];
    end

    % % % % Appending boundary values
    D2(1,1)=1; D2(1,2:N)=0; D2(N+1,1:N)=0; D2(N+1,N+1)=1;

    %% % % % % % % % % % Initialize solution
    u = zeros(N+1, M+1);

    % % % % % % % % Initial condition
    u(:,1) = uexact(x,0);

    % % % % % Define operator Matrix
    B = eye(N+1)-(lambda1)*D2;

    % % % % % % To find the next level solution (i.e, Second level)
    F=u(:,1)+lambda1*[0;f(x(2:N),t(2));0];
    u(:,2)=B\F;
    u(1,2)=b_1(t(2)); u(N+1,2)=b_last(t(2));

    % % % % % % Loop over all the time levels
    for m=2:M
        s=u(:,m);
        for p=m:-1:2
            s=s-a_coeffs(p)*(u(:,m+2-p)-u(:,m+1-p));
        end
        G=s+lambda1*[0;f(x(2:N),t(m+1));0];
        G(1)=b_1(t(m+1)); G(N+1)=b_last(t(m+1));
        u(:,m+1)=B\G;
        u(1,m+1)=b_1(t(m+1)); u(N+1,m+1)=b_last(t(m+1));
    end

    % % % % % % Calculating the Error
    Err_1(k) = norm(uexact(x,t(M+1))-u(:,M+1),inf);

end

%% % % % % % To Calculate the order of convergences
orders_Fd = size(Err_1); % % % % Initialize with zeros for FD
orders = size(Err_1); % % % % Initialize with zeros for RBF-FD
orders_Fd(1) = NaN; orders(1) = NaN;
for j = 2:length(N_values)
    orders_Fd(j) = log(Err_1FD(j-1)/Err_1FD(j))/log(2);
    orders(j) = log(Err_1(j-1)/Err_1(j))/log(2);
end

%% % % % Display results in scientific notation
errors_scientific_FD = sprintfc('%0.4e', Err_1FD); % % % For FD
errors_scientific = sprintfc('%0.4e', Err_1); % % % % For RBF-FD
table(N_values', errors_scientific_FD', orders_Fd',...
    errors_scientific', orders',...
    'VariableNames', {'N', 'FD-Error', 'Cgs. Order-FD', 'RBF-FD Error',...
    'Cgs. Order-RBF-FD'})
