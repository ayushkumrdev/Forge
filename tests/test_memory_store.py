def test_run_lifecycle_and_events(store):
    run_id = store.start_run("add a hello endpoint")
    store.add_event(run_id, "coder", "tool_call", {"tool": "write_file"})
    store.finish_run(run_id, "success", {"changed": ["api.py"]})

    runs = store.recent_runs()
    assert runs[0]["id"] == run_id
    assert runs[0]["status"] == "success"

    events = store.events_for_run(run_id)
    assert len(events) == 1
    assert events[0]["agent"] == "coder"
    assert events[0]["payload"]["tool"] == "write_file"
