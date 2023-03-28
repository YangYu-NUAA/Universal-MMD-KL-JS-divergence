def js_kl(source, target):
    '''
    Calculate the JS/KL divergence and obtain the conditional distribution using joint distribution and marginal distribution
    '''
    

    #Use weight redistribution to calculate the joint distribution of source and target domains, avoiding the inability to calculate due to inconsistent quantities
    def compute_joint_distribution(source_samples, target_samples, num_bins=10):
        n_source = source_samples.shape[0]
        n_target = target_samples.shape[0]
        
        # Compute normalized weights for source and target samples
        source_weights = torch.ones(n_source) / n_source
        target_weights = torch.ones(n_target) / n_target
        
        # Concatenate source and target samples
        joint_samples = torch.cat((source_samples, target_samples))
        joint_weights = torch.cat((source_weights, target_weights))
        
        # Compute joint distribution of source and target samples
        joint_samples_np = joint_samples.cpu().detach().numpy()
        joint_weights_np = joint_weights.cpu().detach().numpy()
        bin_edges = [np.linspace(np.min(joint_samples_np[:,0]), np.max(joint_samples_np[:,0]), num_bins+1), 
                    np.linspace(np.min(joint_samples_np[:,1]), np.max(joint_samples_np[:,1]), num_bins+1)]
        joint_distribution, _ = np.histogramdd(joint_samples_np, bins=bin_edges, weights=joint_weights_np)
        joint_distribution /= np.sum(joint_distribution)
        joint_distribution = torch.tensor(joint_distribution).float()
        
        return joint_distribution

    def compute_marginal_distribution(samples, num_bins=10):
        samples_np = samples.detach().cpu()
        print("device", samples_np.device)  #cpu
        samples_np = samples_np.numpy()

        marginal_distribution, _ = np.histogramdd(samples_np, bins=num_bins, 
                                                range=[(np.min(samples_np[:,0]), np.max(samples_np[:,0])), 
                                                        (np.min(samples_np[:,1]), np.max(samples_np[:,1]))])

        marginal_distribution /= len(samples_np)
     
        marginal_distribution = torch.tensor(marginal_distribution).float()
        
        return marginal_distribution    

    def compute_kl_divergence(p_distribution, q_distribution):

        kl_divergence = F.kl_div(q_distribution.log(),p_distribution)
        #print("kl_divergence",kl_divergence)
        # 
        #kl_divergence = torch.sum(p_distribution * torch.log(p_distribution / q_distribution))
        
        return kl_divergence

    #If the softmax function has been used to convert values between 0-1, there is no need to convert them again.
    source = F.softmax(source, dim = 1)
    target = F.softmax(target, dim = 1)

    source_distribution = compute_marginal_distribution(source)
    
    target_distribution = compute_marginal_distribution(target)

    joint_distribution = compute_joint_distribution(source, target)
    
    # Compute JS divergence using KL divergence
    M = 0.5 * (joint_distribution + source_distribution[:, None] + target_distribution[None, :])

    #Here can obtain two kl divergence
    kl_divergence_source = compute_kl_divergence(source_distribution, M.mean(dim=1))
    kl_divergence_target = compute_kl_divergence(target_distribution, M.mean(dim=0))
    #print("kl",kl_divergence_source)
    
    return 0.5 * (kl_divergence_source + kl_divergence_target)

#Here can obtain js divergence
js = js_kl(distribution_p, distribution_q)
