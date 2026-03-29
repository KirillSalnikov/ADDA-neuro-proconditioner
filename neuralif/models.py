import numml.sparse as sp
import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as pyg
from torch_geometric.nn import aggr
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse import tril

from neuralif.utils import TwoHop, gershgorin_norm


############################
#          Layers          #
############################
class GraphNet(nn.Module):
    # Follows roughly the outline of torch_geometric.nn.MessagePassing()
    # As shown in https://github.com/deepmind/graph_nets
    # Here is a helpful python implementation:
    # https://github.com/NVIDIA/GraphQSat/blob/main/gqsat/models.py
    # Also allows multirgaph GNN via edge_2_features 
    def __init__(self, node_features, edge_features, global_features=0, hidden_size=0,
                 aggregate="mean", activation="relu", skip_connection=False, edge_features_out=None):

        super().__init__()

        # different aggregation functions
        if aggregate == "sum":
            self.aggregate = aggr.SumAggregation()
        elif aggregate == "mean":
            self.aggregate = aggr.MeanAggregation()
        elif aggregate == "max":
            self.aggregate = aggr.MaxAggregation()
        elif aggregate == "softmax":
            self.aggregate = aggr.SoftmaxAggregation(learn=True)
        else:
            raise NotImplementedError(f"Aggregation '{aggregate}' not implemented")

        self.global_aggregate = aggr.MeanAggregation()

        # skip_connection can be bool (legacy: adds 1 feature) or int (number of skip features)
        add_edge_fs = int(skip_connection) if isinstance(skip_connection, bool) else skip_connection
        edge_features_out = edge_features if edge_features_out is None else edge_features_out
        
        # Graph Net Blocks (see https://arxiv.org/pdf/1806.01261.pdf)
        self.edge_block = MLP([global_features + (edge_features + add_edge_fs) + (2 * node_features), 
                               hidden_size,
                               edge_features_out],
                              activation=activation)
        
        self.node_block = MLP([global_features + edge_features_out + node_features,
                               hidden_size,
                               node_features],
                              activation=activation)
        
        # optional set of blocks for global GNN
        self.global_block = None
        if global_features > 0:
            self.global_block = MLP([edge_features_out + node_features + global_features, 
                                     hidden_size,
                                     global_features],
                                    activation=activation)
        
    def forward(self, x, edge_index, edge_attr, g=None):
        row, col = edge_index
        
        if self.global_block is not None:
            assert g is not None, "Need global features for global block"
            
            # run the edge update and aggregate features
            edge_embedding = self.edge_block(torch.cat([torch.ones(x[row].shape[0], 1, device=x.device) * g, 
                                                        x[row], x[col], edge_attr], dim=1))
            aggregation = self.aggregate(edge_embedding, row)
            
            
            agg_features = torch.cat([torch.ones(x.shape[0], 1, device=x.device) * g, x, aggregation], dim=1)
            node_embeddings = self.node_block(agg_features)
            
            # aggregate over all edges and nodes (always mean)
            mp_global_aggr = g
            edge_aggregation_global = self.global_aggregate(edge_embedding)
            node_aggregation_global = self.global_aggregate(node_embeddings)
            
            # compute the new global embedding
            # the old global feature is part of mp_global_aggr
            global_embeddings = self.global_block(torch.cat([node_aggregation_global, 
                                                             edge_aggregation_global,
                                                             mp_global_aggr], dim=1))
            
            return edge_embedding, node_embeddings, global_embeddings
        
        else:
            # update edge features and aggregate
            edge_embedding = self.edge_block(torch.cat([x[row], x[col], edge_attr], dim=1))
            aggregation = self.aggregate(edge_embedding, row)
            agg_features = torch.cat([x, aggregation], dim=1)
            # update node features
            node_embeddings = self.node_block(agg_features)
            return edge_embedding, node_embeddings, None


class MLP(nn.Module):
    def __init__(self, width, layer_norm=False, activation="relu", activate_final=False):
        super().__init__()
        width = list(filter(lambda x: x > 0, width))
        assert len(width) >= 2, "Need at least one layer in the network!"

        lls = nn.ModuleList()
        for k in range(len(width)-1):
            lls.append(nn.Linear(width[k], width[k+1], bias=True))
            if k != (len(width)-2) or activate_final:
                if activation == "relu":
                    lls.append(nn.ReLU())
                elif activation == "tanh":
                    lls.append(nn.Tanh())
                elif activation == "leakyrelu":
                    lls.append(nn.LeakyReLU())
                elif activation == "sigmoid":
                    lls.append(nn.Sigmoid())
                else:
                    raise NotImplementedError(f"Activation '{activation}' not implemented")

        if layer_norm:
            lls.append(nn.LayerNorm(width[-1]))
        
        self.m = nn.Sequential(*lls)

    def forward(self, x):
        return self.m(x)


class MP_Block(nn.Module):
    # L@L.T matrix multiplication graph layer
    # Aligns the computation of L@L.T - A with the learned updates
    def __init__(self, skip_connections, first, last, edge_features, node_features, global_features, hidden_size, **kwargs) -> None:
        super().__init__()

        # first and second aggregation
        if "aggregate" in kwargs and kwargs["aggregate"] is not None:
            aggr = kwargs["aggregate"] if len(kwargs["aggregate"]) == 2 else kwargs["aggregate"] * 2
        else:
            aggr = ["mean", "sum"]

        act = kwargs["activation"] if "activation" in kwargs else "relu"

        input_edge_features = kwargs.get("input_edge_features", 1)
        output_edge_features = kwargs.get("output_edge_features", 1)
        edge_features_in = input_edge_features if first else edge_features
        edge_features_out = output_edge_features if last else edge_features

        # Skip connection adds input_edge_features channels (a_edges concatenated)
        skip_feat = input_edge_features if (not first and skip_connections) else 0

        # We use 2 graph nets in order to operate on the upper and lower triangular parts of the matrix
        self.l1 = GraphNet(node_features=node_features, edge_features=edge_features_in, global_features=global_features,
                           hidden_size=hidden_size, skip_connection=skip_feat,
                           aggregate=aggr[0], activation=act, edge_features_out=edge_features)
        
        self.l2 = GraphNet(node_features=node_features, edge_features=edge_features, global_features=global_features,
                           hidden_size=hidden_size, aggregate=aggr[1], activation=act, edge_features_out=edge_features_out)
    
    def forward(self, x, edge_index, edge_attr, global_features):
        edge_embedding, node_embeddings, global_features = self.l1(x, edge_index, edge_attr, g=global_features)
        
        # flip row and column indices
        edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_embedding, node_embeddings, global_features = self.l2(node_embeddings, edge_index, edge_embedding, g=global_features)
        
        return edge_embedding, node_embeddings, global_features
        

############################
#         Networks         #
############################
class NeuralPCG(nn.Module):
    def __init__(self, **kwargs):
        # NeuralPCG follows the Encoder-Process-Decoder architecture
        super().__init__()
        
        # Network hyper-parameters
        self._latent_size = kwargs["latent_size"]
        self._num_layers = 2
        self._message_passing_steps = kwargs["message_passing_steps"]
        
        # NeuralPCG uses constant number of features for input and output
        self._node_features = 1
        self._edge_features = 1

        # Pre-network transformations
        self.transforms = None
        
        # Encoder - Process - Decoder architecture
        self.encoder_nodes = MLP([self._node_features] + [self._latent_size] * self._num_layers)
        self.encoder_edges = MLP([self._edge_features] + [self._latent_size] * self._num_layers)
        
        # decoder do not have a layer norm
        self.decoder_edges = MLP([self._latent_size] * self._num_layers + [1])
        
        # message passing layers
        self.message_passing = nn.ModuleList([GraphNet(self._latent_size, self._latent_size,
                                                       hidden_size=self._latent_size,
                                                       aggregate="mean")
                                              for _ in range(self._message_passing_steps)])

    def forward(self, data):
        if self.transforms:
            data = self.transforms(data)
        
        x_nodes, x_edges, edge_index = data.x, data.edge_attr, data.edge_index
        
        # save diag elements for later
        diag_idx = edge_index[0] == edge_index[1]
        diag_values = data.edge_attr[diag_idx].clone()

        latent_edges = self.encoder_edges(x_edges)
        latent_nodes = self.encoder_nodes(x_nodes)

        for message_passing_layer in self.message_passing:
            latent_edges, latent_nodes, _ = message_passing_layer(latent_nodes, edge_index, latent_edges)

        # Convert to lower triangular part of a matrix
        decoded_edges = self.decoder_edges(latent_edges)
        
        return self.transform_output_matrix(diag_idx, diag_values, x_nodes, edge_index, decoded_edges)
        
    def transform_output_matrix(self, diag_idx, diag_vals, node_x, edge_index, edge_values):
        # set the diagonal elements
        # the diag element gets duplicated later so we need to divide by 2
        edge_values[diag_idx] = 0.5 * torch.sqrt(diag_vals)
        size = node_x.size()[0]
        
        if torch.is_inference_mode_enabled():
            
            # use scipy to symmetrize output
            m = to_scipy_sparse_matrix(edge_index, edge_values)
            m = m + m.T
            m = tril(m)
            
            # efficient sparse numml format
            l = sp.SparseCSRTensor(m)
            u = sp.SparseCSRTensor(m.T)
            
            return l, u, None
        
        else:
            # symmetrize the output by stacking things!
            transpose_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
            
            sym_value = torch.cat([edge_values, edge_values])
            sym_index = torch.cat([edge_index, transpose_index], dim=1)
            sym_value = sym_value.squeeze()
            
            # return only lower triangular part
            m = sym_index[0] <= sym_index[1]
            
            t = torch.sparse_coo_tensor(sym_index[:, m], sym_value[m],
                                        size=(size, size))
            t = t.coalesce()
            
            return t, None, None


class PreCondNet(nn.Module):
    # BASELINE MODEL
    # No splitting of the matrix into lower and upper part for alignment
    # Used for the ablation study
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
        self.global_features = kwargs["global_features"]
        self.latent_size = kwargs["latent_size"]
        # node features are augmented with local degree profile
        self.augment_node_features = kwargs["augment_nodes"]
        
        num_node_features = 8 if self.augment_node_features else 1
        message_passing_steps = kwargs["message_passing_steps"]
        
        self.skip_connections = kwargs["skip_connections"]
        
        # create the layers
        self.mps = torch.nn.ModuleList()
        for l in range(message_passing_steps):
            self.mps.append(GraphNet(num_node_features,
                                     edge_features=1,
                                     hidden_size=self.latent_size,
                                     skip_connection=(l > 0 and self.skip_connections)))

    def forward(self, data):
        
        if self.augment_node_features:
            data = augment_features(data)
        
        # get the input data
        edge_embedding = data.edge_attr
        node_embedding = data.x
        edge_index = data.edge_index
        
        # copy the input data (only edges of original matrix A) for skip connections
        a_edges = edge_embedding.clone()
        
        # compute the output of the network
        for i, layer in enumerate(self.mps):
            if i != 0 and self.skip_connections:
                edge_embedding = torch.cat([edge_embedding, a_edges], dim=1)
            
            edge_embedding, node_embedding, _ = layer(node_embedding, edge_index, edge_embedding)
        
        # transform the output into a matrix
        return self.transform_output_matrix(node_embedding, edge_index, edge_embedding)
    
    def transform_output_matrix(self, node_x, edge_index, edge_values):
        # force diagonal to be positive (via activation function)
        diag = edge_index[0] == edge_index[1]
        edge_values[diag] = torch.sqrt(torch.exp(edge_values[diag]))
        edge_values = edge_values.squeeze()
        
        size = node_x.size()[0]
        
        if torch.is_inference_mode_enabled():
            # use scipy to symmetrize output
            m = to_scipy_sparse_matrix(edge_index, edge_values)
            m = m + m.T
            m = tril(m)
            
            # efficient sparse numml format
            l = sp.SparseCSRTensor(m)
            u = sp.SparseCSRTensor(m.T)
            
            # Return upper and lower matrix l and u
            return l, u, None
        
        else:
            # symmetrize the output
            # we basicially just stack the indices of the matrix and it's transpose
            # when coalesce the result, these results get summed up.
            transpose_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
            
            sym_value = torch.cat([edge_values, edge_values])
            sym_index = torch.cat([edge_index, transpose_index], dim=1)
            
            # find lower triangular indices
            m = sym_index[0] <= sym_index[1]
            
            # return only lower triangular part of the data
            t = torch.sparse_coo_tensor(sym_index[:, m], sym_value[m], 
                                        size=(size, size))
            
            # take care of duplicate values (to force the output to be symmetric)
            t = t.coalesce()
            
            return t, None, None


class NeuralIF(nn.Module):
    # Neural Incomplete factorization
    def __init__(self, drop_tol=0, **kwargs) -> None:
        super().__init__()

        self.global_features = kwargs["global_features"]
        self.latent_size = kwargs["latent_size"]
        # node features are augmented with local degree profile
        self.augment_node_features = kwargs["augment_nodes"]

        # Complex mode: 2-channel [real, imag] edge features and unit diagonal
        self.complex_mode = kwargs.get("complex_mode", False)
        input_edge_features = 2 if self.complex_mode else 1
        output_edge_features = 2 if self.complex_mode else 1

        num_node_features = 8 if self.augment_node_features else input_edge_features
        message_passing_steps = kwargs["message_passing_steps"]

        # edge feature representation in the latent layers
        edge_features = kwargs.get("edge_features", 1)

        self.skip_connections = kwargs["skip_connections"]

        self.mps = torch.nn.ModuleList()
        for l in range(message_passing_steps):
            # skip connections are added to all layers except the first one
            self.mps.append(MP_Block(skip_connections=self.skip_connections,
                                     first=l==0,
                                     last=l==(message_passing_steps-1),
                                     edge_features=edge_features,
                                     node_features=num_node_features,
                                     global_features=self.global_features,
                                     hidden_size=self.latent_size,
                                     activation=kwargs["activation"],
                                     aggregate=kwargs["aggregate"],
                                     input_edge_features=input_edge_features,
                                     output_edge_features=output_edge_features))

        # node decodings
        self.node_decoder = MLP([num_node_features, self.latent_size, 1]) if kwargs["decode_nodes"] else None

        # diag-aggregation for normalization of rows
        self.normalize_diag = kwargs["normalize_diag"] if "normalize_diag" in kwargs else False
        self.diag_aggregate = aggr.SumAggregation()

        # normalization
        self.graph_norm = pyg.norm.GraphNorm(num_node_features) if ("graph_norm" in kwargs and kwargs["graph_norm"]) else None

        # drop tolerance and additional fill-ins and more sparsity
        self.tau = drop_tol
        self.two = kwargs.get("two_hop", False)
        
    def forward(self, data):
        # ! data could be batched here...(not implemented)
        
        if self.augment_node_features:
            data = augment_features(data, skip_rhs=True)
            
        # add additional edges to the data
        if self.two:
            data = TwoHop()(data)
        
        # * in principle it is possible to integrate reordering here.
        
        data = ToLowerTriangular()(data)
        
        # get the input data
        edge_embedding = data.edge_attr
        l_index = data.edge_index
        
        if self.graph_norm is not None:
            node_embedding = self.graph_norm(data.x, batch=data.batch)
        else:
            node_embedding = data.x
        
        # copy the input data (only edges of original matrix A)
        a_edges = edge_embedding.clone()
        
        if self.global_features > 0:
            global_features = torch.zeros((1, self.global_features), device=data.x.device, requires_grad=False)
            # feature ideas: nnz, 1-norm, inf-norm col/row var, min/max variability, avg distances to nnz
        else:
            global_features = None
        
        # compute the output of the network
        for i, layer in enumerate(self.mps):
            if i != 0 and self.skip_connections:
                edge_embedding = torch.cat([edge_embedding, a_edges], dim=1)
            
            edge_embedding, node_embedding, global_features = layer(node_embedding, l_index, edge_embedding, global_features)
        
        # transform the output into a matrix
        return self.transform_output_matrix(node_embedding, l_index, edge_embedding, a_edges)

    def transform_output_matrix(self, node_x, edge_index, edge_values, a_edges):
        # force diagonal to be positive (real mode) or unit (complex mode)
        diag = edge_index[0] == edge_index[1]

        if self.complex_mode:
            # Complex mode: learnable complex diagonal with guaranteed nonzero magnitude
            # Parameterize L_ii = exp(r/2) * exp(i*theta) where r, theta are network outputs
            # This ensures |L_ii| = exp(r/2) > 0 (non-singular L)
            r = edge_values[diag, 0]
            theta = edge_values[diag, 1]
            amplitude = torch.sqrt(torch.exp(r))  # exp(r/2), always positive
            edge_values[diag, 0] = amplitude * torch.cos(theta)
            edge_values[diag, 1] = amplitude * torch.sin(theta)
        elif self.normalize_diag:
            # copy the diag of matrix A
            a_diag = a_edges[diag]

            # compute the row norm
            square_values = torch.pow(edge_values, 2)
            aggregated = self.diag_aggregate(square_values, edge_index[0])

            # now, we renormalize the edge values such that they are the square root of the original value...
            edge_values = torch.sqrt(a_diag[edge_index[0]]) * edge_values / torch.sqrt(aggregated[edge_index[0]])

        else:
            # otherwise, just take the edge values as they are...
            # but take the square root as it is numerically better
            # edge_values[diag] = torch.exp(edge_values[diag])
            edge_values[diag] = torch.sqrt(torch.exp(edge_values[diag]))

        # node decoder
        node_output = self.node_decoder(node_x).squeeze() if self.node_decoder is not None else None

        # Reconstruct complex values from 2-channel output if in complex mode
        if self.complex_mode:
            matrix_values = torch.complex(edge_values[:, 0], edge_values[:, 1])
        else:
            matrix_values = edge_values.squeeze()

        # ! this if should only be activated when the model is in production!!
        if torch.is_inference_mode_enabled():

            # we can decide to remove small elements during inference from the preconditioner matrix
            if self.tau != 0:
                small_value = (torch.abs(matrix_values) <= self.tau)

                # small value and not diagonal
                elems = torch.logical_and(small_value, torch.logical_not(diag))

                # might be able to do this easily!
                matrix_values[elems] = 0

                # remove zeros from the sparse representation
                filt = (matrix_values != 0)
                matrix_values = matrix_values[filt]
                edge_index = edge_index[:, filt]

            m = torch.sparse_coo_tensor(edge_index, matrix_values,
                                        size=(node_x.size()[0], node_x.size()[0]))

            # produce L and U separately
            l = sp.SparseCSRTensor(m)
            u = sp.SparseCSRTensor(m.T)

            return l, u, node_output

        else:
            # For training and testing (computing regular losses for examples.)
            t = torch.sparse_coo_tensor(edge_index, matrix_values,
                                        size=(node_x.size()[0], node_x.size()[0]))

            # normalized l1 norm
            l1_penalty = torch.sum(torch.abs(matrix_values)) / len(matrix_values)

            return t, l1_penalty, node_output


class LearnedLU(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        
        self.global_features = kwargs["global_features"]
        self.augment_node_features = kwargs["augment_nodes"]
        
        num_node_features = 8 if self.augment_node_features else 1
        
        message_passing_steps = kwargs["message_passing_steps"]
        self.skip_connections = kwargs["skip_connections"]
        self.layers = nn.ModuleList()
        
        # use a smooth activation function for the diagonal during training
        self.smooth_activation = kwargs.get("smooth_activation", True)
        self.epsilon = kwargs.get("epsilon", 0.001)
        
        num_edge_features = 32
        hidden_size = 32
        
        for l in range(message_passing_steps):
            first_layer = l == 0
            last_layer = l == (message_passing_steps - 1)
            
            self.layers.append(
                GraphNet(
                    skip_connection=(l != 0 and self.skip_connections),
                    edge_features=2 if first_layer else num_edge_features,
                    edge_features_out=1 if last_layer else num_edge_features,
                    hidden_size=hidden_size,
                    node_features=num_node_features,
                    global_features=self.global_features
                )
            )
    
    def forward(self, data):
        a_edges = data.edge_attr.clone()

        if self.augment_node_features:
            data = augment_features(data)
        
        # add remaining self loops
        data.edge_index, data.edge_attr = torch_geometric.utils.add_remaining_self_loops(data.edge_index, data.edge_attr)
        
        edge_embedding = data.edge_attr
        node_embedding = data.x
        edge_index = data.edge_index
        
        
        # add positional encoding features
        row, col = data.edge_index
        lower_mask = row > col
        upper_mask = row < col
        additional_edge_feature = torch.zeros_like(a_edges)
        additional_edge_feature[lower_mask] = -1
        additional_edge_feature[upper_mask] = 1
        edge_embedding = torch.cat([edge_embedding, additional_edge_feature], dim=1)
        
        if self.global_features > 0:
            global_features = torch.zeros((1, self.global_features), device=data.x.device, requires_grad=False)
        else:
            global_features = None
        
        for i, layer in enumerate(self.layers):
            if i != 0 and self.skip_connections:
                edge_embedding = torch.cat([edge_embedding, a_edges], dim=1)
                
            edge_embedding, node_embedding, global_features = layer(node_embedding, edge_index, edge_embedding, global_features)
        
        return self.transform_output_matrix(a_edges, node_embedding, edge_index, edge_embedding)
    
    def transform_output_matrix(self, a_edges, node_x, edge_index, edge_values):
        """
        Transform the output into L and U matrices.

        Parameters:
            a_edges (Tensor): Original edge attributes.
            node_x (Tensor): Node features.
            edge_index (Tensor): Edge indices.
            edge_values (Tensor): Edge values.
            tolerance (float): Tolerance for small values.

        Returns:
            tuple: Lower and upper matrices, and L1 norm.
        """
        
        @torch.no_grad()
        def step_activation(x, eps=0.05):
            # activation function to enfore the diagonal to be non-zero
            # - replace small values with epsilon
            # - replace zeros with epsilon
            s = torch.where(torch.abs(x) > eps, x, torch.sign(x) * eps)
            return torch.where(s == 0, eps, s)
            
        def smooth_activation(x, eps=0.05):
            return x * (1 + torch.exp(-torch.abs((4 / eps) * x) + 2))
        
        # create masks to split the edge values
        lower_mask = edge_index[0] >= edge_index[1]
        upper_mask = edge_index[0] <= edge_index[1]
        diag_mask = edge_index[0] == edge_index[1]
        
        # create values and indices for lower part
        lower_indices = edge_index[:, lower_mask]
        lower_values = edge_values[lower_mask][:, 0].squeeze()
        
        # create values and indices for upper part
        upper_indices = edge_index[:, upper_mask]
        upper_values = edge_values[upper_mask][:, -1].squeeze()
        
        # enforce diagonal to be unit valued for the upper part
        upper_values[diag_mask[upper_mask]] = 1
        
        # appy activation function to lower part
        if torch.is_inference_mode_enabled():
            lower_values[diag_mask[lower_mask]] = step_activation(lower_values[diag_mask[lower_mask]], eps=self.epsilon)
        elif self.smooth_activation:
            lower_values[diag_mask[lower_mask]] = smooth_activation(lower_values[diag_mask[lower_mask]], eps=self.epsilon)
        
        # construct L and U matrix
        n = node_x.size()[0]
        
        # convert to lower and upper matrices
        lower_matrix = torch.sparse_coo_tensor(lower_indices, lower_values.squeeze(), size=(n, n))
        upper_matrix = torch.sparse_coo_tensor(upper_indices, upper_values.squeeze(), size=(n, n))
        
        if torch.is_inference_mode_enabled():
            # convert to numml format
            l = sp.SparseCSRTensor(lower_matrix)
            u = sp.SparseCSRTensor(upper_matrix)
            
            return l, u, None
        
        else:
            # min diag element as a regularization term
            bound = torch.min(torch.abs(lower_values[diag_mask[lower_mask]]))
            
            return (lower_matrix, upper_matrix), bound, None


class NeuralSAI(nn.Module):
    """Neural Sparse Approximate Inverse (SAI) preconditioner.

    Predicts a sparse M ≈ A⁻¹ directly (not a factorization).
    Uses left preconditioning: M·A·x = M·b with BiCGStab.

    Architecture: Encoder-Processor-Decoder with extended features.
      - Node features: 9 channels (m_re, m_im, kd, pos, component)
      - Edge features: 8 channels (G_ij, A_ij, direction, 1/kr)
      - Global features: 4 channels (m_re, m_im, kd, log(N))

    Output: sparse M where M = I + M_predicted (residual learning).
    """
    def __init__(self, latent_size=64, message_passing_steps=6,
                 node_features_in=9, edge_features_in=8, global_features_in=4,
                 activation="relu", **kwargs):
        super().__init__()

        self.latent_size = latent_size
        self.message_passing_steps = message_passing_steps
        self.node_features_in = node_features_in
        self.edge_features_in = edge_features_in
        self.global_features_in = global_features_in

        # Encoder
        self.node_enc = MLP([node_features_in, latent_size, latent_size],
                            activation=activation)
        self.edge_enc = MLP([edge_features_in, latent_size, latent_size],
                            activation=activation)
        self.global_enc = MLP([global_features_in, latent_size, latent_size],
                              activation=activation)

        # Processor: K GraphNet layers with skip connections from encoder edge embeddings
        self.processor = nn.ModuleList([
            GraphNet(node_features=latent_size,
                     edge_features=latent_size,
                     global_features=latent_size,
                     hidden_size=latent_size,
                     aggregate="mean",
                     activation=activation)
            for _ in range(message_passing_steps)
        ])

        # Edge skip-connection projection (encoder edge + processor edge → latent)
        self.edge_skip_proj = nn.ModuleList([
            nn.Linear(2 * latent_size, latent_size)
            for _ in range(message_passing_steps)
        ])

        # Decoder
        # Edge decoder: predict [M_ij_re, M_ij_im] correction to identity
        self.edge_dec = MLP([latent_size, latent_size, 2], activation=activation)
        # Diagonal decoder: predict [diag_re, diag_im] correction
        self.diag_dec = MLP([latent_size, latent_size, 2], activation=activation)

    def forward(self, data):
        x_nodes = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Global features: stored as data.global_features [1, 4]
        if hasattr(data, 'global_features') and data.global_features is not None:
            g = data.global_features.squeeze(0)  # [4]
        else:
            g = torch.zeros(self.global_features_in, device=x_nodes.device,
                            dtype=x_nodes.dtype)

        # Encode
        node_latent = self.node_enc(x_nodes)
        edge_latent = self.edge_enc(edge_attr)
        global_latent = self.global_enc(g.unsqueeze(0))  # [1, latent]

        # Save encoder edge embeddings for skip connections
        edge_enc_saved = edge_latent.clone()

        # Process
        for i, (gn_layer, skip_proj) in enumerate(
                zip(self.processor, self.edge_skip_proj)):
            # Skip connection: concatenate encoder edge embeddings
            edge_input = skip_proj(torch.cat([edge_latent, edge_enc_saved], dim=1))
            edge_latent, node_latent, global_latent = gn_layer(
                node_latent, edge_index, edge_input, g=global_latent
            )

        # Decode: edge corrections
        edge_correction = self.edge_dec(edge_latent)  # [n_edges, 2]

        # Diagonal mask
        diag_mask = edge_index[0] == edge_index[1]

        # Diagonal correction from node features
        diag_correction = self.diag_dec(node_latent)  # [n_nodes, 2]

        # Build M values: M = I + correction (residual learning)
        m_re = edge_correction[:, 0].clone()
        m_im = edge_correction[:, 1].clone()

        # Override diagonal with dedicated decoder + identity
        m_re[diag_mask] = 1.0 + diag_correction[edge_index[0][diag_mask], 0]
        m_im[diag_mask] = diag_correction[edge_index[0][diag_mask], 1]

        # Off-diagonal stays as pure correction (I has zeros off-diagonal in graph sparsity)

        n = x_nodes.size(0)
        complex_values = torch.complex(m_re, m_im)

        if torch.is_inference_mode_enabled():
            # Return sparse CSR for efficient SpMV
            m_coo = torch.sparse_coo_tensor(
                edge_index, complex_values, size=(n, n)
            ).coalesce()

            # Convert to scipy CSR, then to numml or keep as torch sparse
            # For ADDA export, we'll convert externally
            m_csr = m_coo.to_sparse_csr()
            return m_csr, None, None
        else:
            # Training mode: return COO + L1 penalty
            m_sparse = torch.sparse_coo_tensor(
                edge_index, complex_values, size=(n, n)
            ).coalesce()

            # L1 penalty on off-diagonal elements (encourage sparsity)
            off_diag_mask = ~diag_mask
            l1_penalty = torch.sum(torch.abs(complex_values[off_diag_mask])) / max(off_diag_mask.sum().item(), 1)

            return m_sparse, l1_penalty, None


############################
#         HELPERS          #
############################
def augment_features(data, skip_rhs=False):
    # transform nodes to include more features
    
    if skip_rhs:
        # use instead notde position as an input feature!
        data.x = torch.arange(data.x.size()[0], device=data.x.device).unsqueeze(1)
    
    data = torch_geometric.transforms.LocalDegreeProfile()(data)
    
    # diagonal dominance and diagonal decay from the paper
    row, col = data.edge_index
    diag = (row == col)
    diag_elem = torch.abs(data.edge_attr[diag])
    # remove diagonal elements by setting them to zero
    non_diag_elem = data.edge_attr.clone()
    non_diag_elem[diag] = 0
    
    row_sums = aggr.SumAggregation()(torch.abs(non_diag_elem), row)
    alpha = diag_elem / row_sums
    row_dominance_feature = alpha / (alpha + 1)
    row_dominance_feature = torch.nan_to_num(row_dominance_feature, nan=1.0)
    
    # compute diagonal decay features
    row_max = aggr.MaxAggregation()(torch.abs(non_diag_elem), row)
    alpha = diag_elem / row_max
    row_decay_feature = alpha / (alpha + 1)
    row_decay_feature = torch.nan_to_num(row_decay_feature, nan=1.0)
    
    data.x = torch.cat([data.x, row_dominance_feature, row_decay_feature], dim=1)
    
    return data
    
class ToLowerTriangular:
    def __init__(self, inplace=False):
        self.inplace = inplace
        
    def __call__(self, data, order=None):
        if not self.inplace:
            data = data.clone()
        
        # transform the data into lower triangular graph
        rows, cols = data.edge_index[0], data.edge_index[1]
        fil = cols <= rows
        l_index = data.edge_index[:, fil]
        edge_embedding = data.edge_attr[fil]
        
        data.edge_index, data.edge_attr = l_index, edge_embedding
        return data
