from rag.eval.throttle import Throttle


def test_calls_are_spaced_by_min_interval():
    now = [100.0]
    sleeps: list[float] = []

    def sleep(s):
        sleeps.append(s)
        now[0] += s

    t = Throttle(2.0, clock=lambda: now[0], sleep=sleep)
    t.wait()                      # first call: no wait
    t.wait()                      # second: wait the full interval
    now[0] += 5                   # idle longer than the interval
    t.wait()                      # no wait needed
    assert sleeps == [2.0]


def test_zero_interval_never_sleeps():
    t = Throttle(0.0, clock=lambda: 0.0, sleep=lambda s: (_ for _ in ()).throw(AssertionError("slept")))
    for _ in range(3):
        t.wait()
