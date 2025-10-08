import anyio

from lambdapic.cli.mcp import (
    DocumentationIndex,
    get_doc,
    get_code,
    list_callbacks,
    list_simulations,
    mcp_app,
)


def test_documentation_index_finds_simulations():
    index = DocumentationIndex()

    qualified_names = {record.qualified_name for record in index.simulations}
    assert "lambdapic.simulation.Simulation" in qualified_names
    assert "lambdapic.simulation.Simulation3D" in qualified_names

    doc = index.get_doc("lambdapic.simulation.Simulation")
    assert doc is not None
    assert "Particle-In-Cell" in doc


def test_documentation_index_finds_callbacks():
    index = DocumentationIndex()

    callbacks = {record.qualified_name: record for record in index.callbacks}
    assert "lambdapic.callback.hdf5.SaveFieldsToHDF5" in callbacks
    assert callbacks["lambdapic.callback.hdf5.SaveFieldsToHDF5"].stage == "maxwell second"

    doc = index.get_doc("lambdapic.callback.hdf5.SaveFieldsToHDF5")
    assert doc is not None
    assert "Callback to save field data" in doc


def test_fastmcp_registers_expected_tools():
    tools = {tool.name for tool in mcp_app._tool_manager.list_tools()}
    assert {"list_simulations", "list_callbacks", "get_doc", "get_code"} <= tools


def test_documentation_index_returns_source():
    index = DocumentationIndex()

    source = index.get_source("lambdapic.simulation.Simulation")
    assert source is not None
    assert "class Simulation(" in source or "class Simulation:" in source


def test_list_simulations_tool_text_contains_known_class():
    text = list_simulations()
    assert "Available simulation classes" in text
    assert "lambdapic.simulation.Simulation" in text


def test_list_callbacks_tool_text_contains_known_callback():
    text = list_callbacks()
    assert "Available callbacks" in text
    assert "lambdapic.callback.hdf5.SaveFieldsToHDF5" in text


def test_get_doc_tool_returns_docstring():
    text = get_doc("lambdapic.simulation.Simulation")
    assert "Particle-In-Cell" in text


def test_get_code_tool_returns_source():
    text = get_code("lambdapic.simulation.Simulation")
    assert "class Simulation(" in text or "class Simulation:" in text


def test_manual_resource_available_and_readable():
    manual_uri = "doc://lambdapic/manual"
    resources = {str(resource.uri): resource for resource in mcp_app._resource_manager.list_resources()}
    assert manual_uri in resources

    async def _read():
        resource = await mcp_app._resource_manager.get_resource(manual_uri)
        assert resource is not None
        content = await resource.read()
        return content

    content = anyio.run(_read)
    assert "LambdaPIC MCP Manual" in content
    assert "Authoring a Simulation Script" in content
