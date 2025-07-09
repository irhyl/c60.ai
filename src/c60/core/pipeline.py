import uuid
import networkx as nx
from typing import Any, Dict, List, Optional, Callable, Union

class PipelineStep:
    """
    Represents a single atomic operation or 'step' within an ML pipeline.
    This class acts as a node in our molecular graph representation.
    Each step encapsulates a specific callable operation (e.g., a transformer,
    an estimator, or a custom function) along with its unique identifier
    and configuration parameters.
    """
    def __init__(
        self,
        step_id: Optional[str] = None,
        name: str = "UnnamedStep",
        step_type: str = "generic", # e.g., 'transformer', 'estimator', 'feature_engineer', 'custom'
        operation: Optional[Callable[..., Any]] = None, # The actual callable object for this step
        params: Optional[Dict[str, Any]] = None # Hyperparameters for the operation
    ):
        """
        Initializes a PipelineStep instance.

        Args:
            step_id (Optional[str]): A unique identifier for this step. If not provided,
                                      a UUID will be automatically generated.
            name (str): A human-readable name for the step, useful for debugging and visualization.
            step_type (str): A categorical label for the step's function (e.g., 'transformer',
                             'estimator', 'data_source', 'feature_engineer').
            operation (Optional[Callable[..., Any]]): The actual Python callable that this
                                                      step performs. This could be an instance
                                                      of an `sklearn` transformer/model, or a
                                                      custom function.
            params (Optional[Dict[str, Any]]): A dictionary of parameters to be passed to
                                              the `operation` when it's executed or initialized.
        """
        # Assign a unique ID, generating one if not provided. This ensures distinct nodes in the graph.
        self.id: str = step_id if step_id is not None else str(uuid.uuid4())
        self.name: str = name
        self.step_type: str = step_type
        self.operation: Optional[Callable[..., Any]] = operation
        # Ensure params is always a dictionary, defaulting to empty if None
        self.params: Dict[str, Any] = params if params is not None else {}

    def __repr__(self) -> str:
        """
        Provides a concise string representation of the PipelineStep for debugging purposes.
        Shows a truncated ID, name, and type.
        """
        return f"PipelineStep(id='{self.id[:8]}...', name='{self.name}', type='{self.step_type}')"

    def __eq__(self, other: Any) -> bool:
        """
        Defines equality for PipelineStep objects based on their unique `id`.
        This is crucial for correct graph operations and set/dictionary usage.
        """
        if not isinstance(other, PipelineStep):
            return NotImplemented # Allow comparison with other types to return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """
        Defines the hash for PipelineStep objects based on their unique `id`.
        Required for objects to be used in sets or as dictionary keys.
        """
        return hash(self.id)

class Pipeline:
    """
    Represents an entire Machine Learning pipeline as a Directed Acyclic Graph (DAG).
    This is the 'molecular' structure in C60.ai, where nodes are PipelineSteps
    and edges represent the flow of data. It uses the networkx library for graph management.
    """
    def __init__(self, name: str = "UnnamedPipeline"):
        """
        Initializes a new Pipeline instance.

        Args:
            name (str): A human-readable name for the pipeline.
        """
        self.name: str = name
        # Initialize a directed graph using networkx. This will store our PipelineStep objects.
        self.graph: nx.DiGraph = nx.DiGraph()
        # Placeholders for special input/output nodes. These will be implemented in later phases
        # to explicitly mark data entry and exit points for the entire pipeline.
        self._input_node_id: Optional[str] = None
        self._output_node_id: Optional[str] = None

    def add_step(self, step: PipelineStep) -> None:
        """
        Adds a PipelineStep as a node to the pipeline graph.
        Each node in the networkx graph stores the PipelineStep object itself as 'data'.

        Args:
            step (PipelineStep): The PipelineStep instance to add to the graph.

        Raises:
            ValueError: If a step with the same ID already exists in the pipeline,
                        preventing duplicate nodes.
        """
        if self.graph.has_node(step.id):
            raise ValueError(f"Error: Step with ID '{step.id}' already exists in the pipeline. "
                             "Each pipeline step must have a unique ID.")
        # Add the node to the networkx graph, storing the PipelineStep object in the 'data' attribute.
        self.graph.add_node(step.id, data=step)

    def add_edge(self, from_step_id: str, to_step_id: str) -> None:
        """
        Adds a directed edge (representing data flow) between two existing steps in the pipeline.
        This method includes crucial validation to ensure the graph remains a DAG.

        Args:
            from_step_id (str): The ID of the source PipelineStep (data producer).
            to_step_id (str): The ID of the destination PipelineStep (data consumer).

        Raises:
            ValueError: If either the source or destination step ID does not exist in the graph,
                        or if adding the edge would create a cycle, violating the DAG property.
        """
        # Validate that both source and destination nodes exist in the graph
        if not self.graph.has_node(from_step_id):
            raise ValueError(f"Error: Source step with ID '{from_step_id}' not found in pipeline. "
                             "Ensure steps are added before creating edges.")
        if not self.graph.has_node(to_step_id):
            raise ValueError(f"Error: Destination step with ID '{to_step_id}' not found in pipeline. "
                             "Ensure steps are added before creating edges.")

        # Add the edge. Networkx allows adding existing edges without error.
        self.graph.add_edge(from_step_id, to_step_id)

        # Crucial DAG validation: Check for cycles immediately after adding an edge.
        # If a cycle is detected, the edge is removed to revert to a valid state.
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(from_step_id, to_step_id) # Rollback the invalid edge addition
            raise ValueError(f"Error: Adding edge from '{from_step_id}' to '{to_step_id}' would create a cycle. "
                             "Pipelines must be Directed Acyclic Graphs (DAGs).")

    def get_step(self, step_id: str) -> PipelineStep:
        """
        Retrieves a PipelineStep object from the graph by its unique ID.

        Args:
            step_id (str): The ID of the step to retrieve.

        Returns:
            PipelineStep: The requested PipelineStep object.

        Raises:
            KeyError: If no step with the given ID is found in the pipeline.
        """
        try:
            # Access the 'data' attribute of the node, where the PipelineStep object is stored.
            return self.graph.nodes[step_id]['data']
        except KeyError:
            raise KeyError(f"Error: Step with ID '{step_id}' not found in pipeline.")

    def get_nodes(self) -> List[PipelineStep]:
        """
        Returns a list of all PipelineStep objects (nodes) currently in the pipeline.
        The order is not guaranteed to be topological.
        """
        return [self.graph.nodes[node_id]['data'] for node_id in self.graph.nodes]

    def get_edges(self) -> List[tuple[str, str]]:
        """
        Returns a list of all edges in the pipeline. Each edge is represented as a tuple
        (source_step_id, destination_step_id).
        """
        return list(self.graph.edges)

    def get_sources(self) -> List[PipelineStep]:
        """
        Identifies and returns a list of PipelineStep objects that are 'source' nodes
        in the graph (i.e., nodes with no incoming edges). These are typically the
        entry points for data into the pipeline.
        """
        return [self.graph.nodes[node_id]['data'] for node_id in self.graph.nodes if self.graph.in_degree(node_id) == 0]

    def get_sinks(self) -> List[PipelineStep]:
        """
        Identifies and returns a list of PipelineStep objects that are 'sink' nodes
        in the graph (i.e., nodes with no outgoing edges). These typically represent
        the final outputs or models in a pipeline.
        """
        return [self.graph.nodes[node_id]['data'] for node_id in self.graph.nodes if self.graph.out_degree(node_id) == 0]

    def __len__(self) -> int:
        """
        Returns the total number of steps (nodes) currently in the pipeline.
        """
        return self.graph.number_of_nodes()

    def __repr__(self) -> str:
        """
        Provides a concise string representation of the Pipeline object,
        showing its name and the count of steps and edges.
        """
        return (f"Pipeline(name='{self.name}', steps={len(self)}, "
                f"edges={self.graph.number_of_edges()})")

    def __str__(self) -> str:
        """
        Provides a more detailed, human-readable string representation of the Pipeline,
        listing the names of its steps and all its edges.
        """
        node_names = [step.name for step in self.get_nodes()]
        return (f"Pipeline: {self.name}\n"
                f"  Steps: {', '.join(node_names) if node_names else 'None'}\n"
                f"  Edges: {self.get_edges() if self.get_edges() else 'None'}")
